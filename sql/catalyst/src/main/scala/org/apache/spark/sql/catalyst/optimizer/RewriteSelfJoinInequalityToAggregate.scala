/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.catalyst.optimizer

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{Max, Min}
import org.apache.spark.sql.catalyst.plans.{Inner, Repeatability}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.IN_SUBQUERY
import org.apache.spark.sql.catalyst.util.UnsafeRowUtils
import org.apache.spark.sql.internal.SQLConf

/**
 * Rewrites an existence-only self-join of the form
 *
 *   R l JOIN R r
 *     ON l.k1 = r.k1 [AND ...]
 *    AND l.v <> r.v
 *
 * into a decomposable MIN/MAX aggregation over one copy of R:
 *
 *   SELECT k1 [, ...]
 *   FROM R
 *   WHERE k1 IS NOT NULL [AND ...]
 *   GROUP BY k1 [, ...]
 *   HAVING MIN(v) <> MAX(v)
 *
 * The rewrite is valid only when the multiplicity produced by the self-join is unobservable.
 * This rule therefore operates before [[RewritePredicateSubquery]] and only inside uncorrelated
 * IN subqueries. Non-correlated EXISTS has already been rewritten by [[RewriteNonCorrelatedExists]]
 * in `FinishAnalysis` before this rule runs, so EXISTS is intentionally outside this rule's
 * matching surface.
 *
 * Non-binary-equality collations are rejected explicitly. This keeps correctness local to this
 * rule instead of relying on the current behavior of `RewriteCollationJoin` to make such
 * predicates syntactically unmatchable. Floating-point equi keys are also rejected because Spark
 * equality normalizes signed zero and NaN representations while projected expressions may observe
 * the original representation.
 *
 * The initial rule handles the direct logical shape:
 *
 *   Project
 *     +- Inner self-join
 *
 * Pre-existing LeftSemi/LeftAnti plans, EXISTS/ScalarSubquery shapes, nested self-joins under
 * another join, and Aggregate/Distinct wrappers above the candidate join are intentionally not
 * rewritten. Project expressions are supported when every reference to the eliminated self-join
 * can be remapped to an equality key that survives the aggregation.
 *
 * The implementation deliberately fails closed. In particular it requires:
 *   - an uncorrelated IN context;
 *   - exactly one inequality pair and at least one equality-key pair;
 *   - both sides of the self-join to be the same logical relation, and that relation to be
 *     repeatable, since replacing two evaluations by one is unsound otherwise: every leaf must
 *     either declare its output repeatable through
 *     [[org.apache.spark.sql.catalyst.plans.logical.RepeatableLeafNode]] or be one of Catalyst's
 *     own pure leaves;
 *   - every expression in the subquery, not only those in the relation, to stay inside the trust
 *     boundary [[org.apache.spark.sql.catalyst.plans.Repeatability]] defines, because
 *     `deterministic` alone does not promise that two evaluations agree, and expressions above the
 *     eliminated join are evaluated once per group instead of once per matching pair;
 *   - no embedded subquery in either relation being folded from two evaluations into one;
 *   - every equality/inequality pair to reference corresponding output slots from the two sides;
 *   - equality-key types not to require floating-point normalization;
 *   - no hint on the self-join being eliminated;
 *   - every projected output that depends on the self-join to be remappable to an equality key.
 *
 * Attribute correspondence is established by output position after proving that the two child
 * plans are canonically equal. Column names are never used as attribute identity.
 *
 * The rule intentionally runs immediately before [[RewritePredicateSubquery]], after subqueries
 * have already been optimized, because it needs to observe the `InSubquery` logical shape before
 * Spark converts it to a semi/anti/existence join.
 *
 * MIN/MAX is used instead of COUNT(DISTINCT) so the replacement remains a normal decomposable
 * aggregation with no distinct-shuffle stage. The inequality value must be orderable because both
 * MIN and MAX use Spark's ordering semantics.
 */
object RewriteSelfJoinInequalityToAggregate
    extends Rule[LogicalPlan]
    with PredicateHelper {

  private val MinAliasName = "_self_join_min"
  private val MaxAliasName = "_self_join_max"

  private case class RelationCorrespondence(leftToRight: AttributeMap[Attribute]) {
    def areCorresponding(left: Attribute, right: Attribute): Boolean =
      leftToRight.get(left).exists(_.exprId == right.exprId)
  }

  private case class SelfJoinSpec(
      equiPairs: Seq[(Attribute, Attribute)],
      neqPair: (Attribute, Attribute)) {
    def leftEquiKeys: Seq[Attribute] = equiPairs.map(_._1)
    def leftNeq: Attribute = neqPair._1
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!conf.getConf(SQLConf.REWRITE_SELF_JOIN_INEQUALITY_TO_AGGREGATE_ENABLED)) {
      return plan
    }

    plan.transformAllExpressionsWithPruning(_.containsPattern(IN_SUBQUERY)) {
      case in @ InSubquery(_, query: ListQuery) if query.children.isEmpty =>
        rewriteSubquery(query.plan) match {
          case Some(newPlan) => in.copy(query = query.withNewPlan(newPlan))
          case None => in
        }
    }
  }

  private def rewriteSubquery(plan: LogicalPlan): Option[LogicalPlan] = {
    if (!isCandidateSafe(plan)) return None

    plan match {
      case project @ Project(_, selfJoin: Join) if isInnerJoin(selfJoin) =>
        extractSelfJoin(selfJoin).flatMap { spec =>
          rewriteDirectSelfJoin(project, selfJoin, spec)
        }
      case _ =>
        None
    }
  }

  // --------------------------------------------------------------------------
  // Direct self-join in the subquery.
  // --------------------------------------------------------------------------

  private def rewriteDirectSelfJoin(
      project: Project,
      selfJoin: Join,
      spec: SelfJoinSpec): Option[LogicalPlan] = {
    // IN consumes ListQuery output positionally. A bare self-join exposes both sides' full
    // schemas, so replacing it with only equality keys would change arity/position. Require a
    // Project whose output can be remapped safely while preserving its positional schema.
    // `rewriteKeyProject` takes its new child by name, so the aggregation is only built once the
    // whole project list has been proven remappable.
    rewriteKeyProject(
      project.projectList,
      spec,
      selfJoin.outputSet,
      buildAggregate(spec.leftEquiKeys, spec.leftNeq, selfJoin.left))
  }

  // --------------------------------------------------------------------------
  // Self-join recognition and predicate proof.
  // --------------------------------------------------------------------------

  private def extractSelfJoin(join: Join): Option[SelfJoinSpec] = {
    if (!isInnerJoin(join) || !join.hint.isEmpty) {
      None
    } else {
      for {
        correspondence <- buildCorrespondence(join.left, join.right)
        spec <- parseCondition(
          join.condition.get,
          join.left.outputSet,
          join.right.outputSet,
          correspondence)
      } yield spec
    }
  }

  private def buildCorrespondence(
      left: LogicalPlan,
      right: LogicalPlan): Option[RelationCorrespondence] = {
    if (!sameRepeatableRelation(left, right) || left.output.size != right.output.size) {
      None
    } else {
      val pairs = left.output.zip(right.output)
      if (!pairs.forall { case (l, r) => l.dataType == r.dataType }) {
        None
      } else {
        Some(RelationCorrespondence(leftToRight = AttributeMap(pairs)))
      }
    }
  }

  private def parseCondition(
      condition: Expression,
      leftOutput: AttributeSet,
      rightOutput: AttributeSet,
      correspondence: RelationCorrespondence): Option[SelfJoinSpec] = {
    // ExpressionSet deduplicates semantically equal conjuncts while preserving their order, which
    // keeps the exhaustive-coverage count below from rejecting a condition that merely repeats a
    // predicate.
    val predicates = ExpressionSet(splitConjunctivePredicates(condition)).toSeq

    def orient(a: Attribute, b: Attribute): Option[(Attribute, Attribute)] = {
      if (leftOutput.contains(a) && rightOutput.contains(b)) {
        Some((a, b))
      } else if (leftOutput.contains(b) && rightOutput.contains(a)) {
        Some((b, a))
      } else {
        None
      }
    }

    val equiPairs = predicates.flatMap {
      case EqualTo(leftExpr, rightExpr) =>
        for {
          leftAttr <- extractJoinAttribute(leftExpr)
          rightAttr <- extractJoinAttribute(rightExpr)
          pair <- orient(leftAttr, rightAttr)
          if correspondence.areCorresponding(pair._1, pair._2)
        } yield pair
      case _ => None
    }

    val neqPairs = predicates.flatMap {
      case Not(EqualTo(a: Attribute, b: Attribute)) =>
        orient(a, b).filter { case (l, r) => correspondence.areCorresponding(l, r) }
      case _ => None
    }

    val joinAttrIds = (equiPairs ++ neqPairs)
      .flatMap { case (l, r) => Seq(l.exprId, r.exprId) }.toSet
    val acceptedIsNotNull = predicates.count {
      case IsNotNull(expr) =>
        extractJoinAttribute(expr).exists(a => joinAttrIds.contains(a.exprId))
      case _ => false
    }

    // Every conjunct must have been proven safe. This also rejects EqualNullSafe, OR predicates,
    // computed join expressions, additional residual predicates and non-corresponding columns.
    if (equiPairs.size + neqPairs.size + acceptedIsNotNull != predicates.size) return None
    if (equiPairs.isEmpty || neqPairs.size != 1) return None

    val leftEquiIds = equiPairs.map(_._1.exprId)
    val neqPair = neqPairs.head
    if (leftEquiIds.contains(neqPair._1.exprId)) return None

    val equiKeyTypes = equiPairs.map(_._1.dataType)

    // Equality keys whose values Spark normalizes cannot be substituted inside an arbitrary
    // projected expression: -0.0 and 0.0 compare equal as join and grouping keys, yet an
    // expression over their original representations tells them apart, and this rule may
    // substitute one side's key. Fail closed for exactly the types Spark itself considers
    // normalization-sensitive. The inequality value is not substituted and may still be
    // floating point.
    if (equiKeyTypes.exists(NormalizeFloatingNumbers.needNormalize)) return None

    // Keep collation/equality semantics local to this rule instead of relying on
    // RewriteCollationJoin making unsupported predicates syntactically unmatchable.
    val comparedTypes = equiKeyTypes :+ neqPair._1.dataType
    if (!comparedTypes.forall(UnsafeRowUtils.isBinaryStable)) return None

    // MIN/MAX accept any Spark-orderable type, and so does GROUP BY, so require orderability for
    // the grouping keys the rewrite introduces as well as for the aggregated inequality value. The
    // rewrite theorem relies on the ordering's equivalence classes agreeing with equality, while
    // the binary-stability guard above excludes non-binary collation semantics that we
    // intentionally do not prove here. EqualTo already requires an orderable type, so this is
    // defensive.
    if (!comparedTypes.forall(RowOrdering.isOrderable)) return None

    Some(SelfJoinSpec(equiPairs, neqPair))
  }

  private def extractJoinAttribute(expression: Expression): Option[Attribute] = expression match {
    case a: Attribute => Some(a)
    case _ => None
  }

  // --------------------------------------------------------------------------
  // Rewrite builders and ExprId remapping.
  // --------------------------------------------------------------------------

  private def buildAggregate(
      equiKeys: Seq[Attribute],
      neq: Attribute,
      child: LogicalPlan): LogicalPlan = {
    // `l.k = r.k` never matches when either equi key is NULL, while GROUP BY would otherwise
    // form a NULL group. Emitting such a group changes semantics, most visibly for NOT IN: a NULL
    // key in the subquery can make the null-aware anti join reject every outer row. Filter NULL
    // equi keys before aggregation to preserve the original self-join semantics.
    //
    // MIN/MAX ignore NULL inequality values. If a group has fewer than two distinct non-NULL
    // values, MIN(v) and MAX(v) compare equal (or are both NULL), so the Filter rejects it. If the
    // group has at least two distinct non-NULL values, the minimum and maximum compare unequal.
    // InferFiltersFromConstraints has already run before this late optimizer batch. Avoid adding
    // duplicate IsNotNull predicates that are already implied by the child; there is no later
    // PruneFilters fixed point that would reliably clean them up.
    val missingNonNullPredicates = equiKeys
      .map(a => IsNotNull(a): Expression)
      .filterNot(child.constraints.contains)
    val nonNullCondition = missingNonNullPredicates.reduceOption(And)
    val filteredChild = nonNullCondition.map(Filter(_, child)).getOrElse(child)

    val minAlias = Alias(Min(neq).toAggregateExpression(), MinAliasName)()
    val maxAlias = Alias(Max(neq).toAggregateExpression(), MaxAliasName)()
    val aggregate = Aggregate(equiKeys, equiKeys ++ Seq(minAlias, maxAlias), filteredChild)

    Filter(Not(EqualTo(minAlias.toAttribute, maxAlias.toAttribute)), aggregate)
  }

  /**
   * Rebuild a Project after eliminating a self-join.
   *
   * Bare equality-key attributes receive fresh output identities because two positional slots such
   * as `l.k` and `r.k` must not collapse to the same ExprId after both are backed by the surviving
   * left key. Alias outputs already have their own identities, so their ExprIds are preserved while
   * references in the Alias child are remapped. Foldable outputs are row-invariant and can be
   * retained unchanged.
   *
   * `newChild` is taken by name so that the replacement plan is only built after the whole project
   * list has been proven remappable. It is forced exactly once, at the single binding below,
   * because a plan builder mints fresh ExprIds per call and evaluating it twice would produce two
   * different trees.
   */
  private def rewriteKeyProject(
      projectList: Seq[NamedExpression],
      spec: SelfJoinSpec,
      eliminatedOutput: AttributeSet,
      newChild: => LogicalPlan): Option[Project] = {
    val keyRemap = AttributeMap(
      spec.leftEquiKeys.map(a => a -> a) ++ spec.equiPairs.map { case (l, r) => r -> l })

    rewriteProjectList(projectList, keyRemap, eliminatedOutput).map { rewritten =>
      val child = newChild
      Project(rewritten, child)
    }
  }

  /**
   * Remap references to an eliminated subtree inside a Project.
   *
   * References to attributes outside `eliminatedOutput` are left untouched.
   */
  private def rewriteProjectList(
      projectList: Seq[NamedExpression],
      remap: AttributeMap[Attribute],
      eliminatedOutput: AttributeSet): Option[Seq[NamedExpression]] = {
    // SubqueryExpression.references only describes outer references; expression transforms do not
    // rewrite the subquery plan itself. Fail closed rather than risk leaving an OuterReference in
    // that plan pointing at an eliminated ExprId.
    if (projectList.exists(SubqueryExpression.hasSubquery)) {
      return None
    }

    val rewritten = projectList.map {
      case a: Attribute if remap.contains(a) =>
        // A bare Attribute has no independent output identity. Always create a new slot so two
        // old slots that now read the same surviving key still have distinct ExprIds.
        Some(Alias(remap(a), a.name)(qualifier = a.qualifier): NamedExpression)

      case a: Attribute if eliminatedOutput.contains(a) =>
        None

      case a: Attribute =>
        Some(a: NamedExpression)

      case al: Alias =>
        val eliminatedRefs = al.references.filter(eliminatedOutput.contains)
        if (!eliminatedRefs.forall(remap.contains)) {
          None
        } else if (eliminatedRefs.nonEmpty) {
          val newChild = al.child.transformUp {
            case a: Attribute if remap.contains(a) => remap(a)
          }
          Some(al.withNewChild(newChild): NamedExpression)
        } else if (al.references.isEmpty && !al.child.foldable) {
          // Zero-reference outputs must be row-invariant. Alias itself is not foldable, so test
          // the child (e.g. Alias(Literal(1), ...)).
          None
        } else {
          Some(al: NamedExpression)
        }

      // Defensive fallback for future NamedExpression implementations: never retain an expression
      // that still depends on an attribute from the eliminated subtree.
      case other if other.references.exists(eliminatedOutput.contains) =>
        None

      case other if other.references.isEmpty && !other.foldable =>
        None

      case other =>
        Some(other)
    }

    if (rewritten.exists(_.isEmpty)) None else Some(rewritten.flatten)
  }

  // --------------------------------------------------------------------------
  // Candidate safety / repeatability / same-relation proof.
  // --------------------------------------------------------------------------

  private def isInnerJoin(join: Join): Boolean =
    join.joinType == Inner && join.condition.isDefined

  /**
   * Whether the IN subquery as a whole is a safe candidate.
   *
   * Being deterministic and non-streaming is necessary but not sufficient. Collapsing the
   * self-join's multiplicity means every expression above the eliminated join -- the surviving
   * Project, an enclosing join condition -- is evaluated once per group where it used to be
   * evaluated once per matching pair. A genuinely deterministic expression cannot observe that,
   * since the same input yields the same value, but a deterministic-looking one can: `aes_encrypt`
   * draws a fresh initialization vector on every evaluation, and a user-defined or otherwise opaque
   * expression may do anything at all. Emitting one value where the self-join emitted several would
   * then change the value set the enclosing IN compares against, which is why the whole subquery is
   * required to stay inside the trust boundary [[Repeatability]] defines.
   *
   * Checking the whole subquery is deliberately stronger than that argument needs: a subtree beside
   * the candidate, such as the other side of an enclosing join, keeps its own row count and could
   * be exempted. Failing closed there is not worth a carve-out.
   *
   * This is the expression half of repeatability, and the only place the rule asks about
   * expressions: `isRepeatableRelation` builds on it and adds the operator and leaf half.
   */
  private def isCandidateSafe(plan: LogicalPlan): Boolean = {
    plan.deterministic &&
      !plan.isStreaming &&
      !plan.exists(node => !Repeatability.hasSafeExpressions(node))
  }

  private def sameRepeatableRelation(left: LogicalPlan, right: LogicalPlan): Boolean =
    left.sameResult(right) && isRepeatableRelation(left) && isRepeatableRelation(right)

  /**
   * Whether reading `plan` twice returns the same rows, which is what makes folding the two sides
   * of the self-join into one evaluation sound. `isCandidateSafe` answers the expression half of
   * that question; the operators and leaves are checked below.
   */
  private def isRepeatableRelation(plan: LogicalPlan): Boolean = {
    isCandidateSafe(plan) &&
      plan.subqueriesAll.isEmpty &&
      !containsNonRepeatableOperator(plan)
  }

  private def containsNonRepeatableOperator(plan: LogicalPlan): Boolean = plan.exists {
    // Leaf relations fail closed, because `deterministic` is not evidence that two evaluations
    // return the same rows: a leaf commonly has no expressions at all, so a relation backed by an
    // arbitrary RDD or connector is trivially deterministic while its contents may change between
    // reads. Only a leaf that declares its output repeatable is accepted, plus the Catalyst leaves
    // that carry their rows in the plan or generate them from constants.
    //
    // This also rejects CTE and recursive-loop references, whose definitions are not visible from
    // this subtree and may contain LIMIT, Sample, nondeterminism or recursion; the
    // two-evaluations-to-one-evaluation proof must not depend on later CTE replacement or on
    // exchange reuse.
    case leaf: RepeatableLeafNode => !leaf.isOutputRepeatable
    case _: LocalRelation | _: OneRowRelation | _: Range => false
    case _: LeafNode => true
    case _: Project => false
    case _: Filter => false
    case _: SubqueryAlias => false
    case j: Join if j.joinType == Inner => false
    case _ => true
  }
}
