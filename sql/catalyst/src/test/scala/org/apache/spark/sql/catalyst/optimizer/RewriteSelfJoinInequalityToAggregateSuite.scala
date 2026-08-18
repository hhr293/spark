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

import java.nio.charset.StandardCharsets

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.dsl.expressions._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{Max, Min}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.plans.{Inner, LeftOuter, PlanTest, QueryPlan}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.RuleExecutor
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, MapType, StringType, StructField, StructType}

/**
 * Plan-level tests for [[RewriteSelfJoinInequalityToAggregate]]: the matcher, the fail-closed
 * boundary, `ExprId` hygiene, the enclosing-join shape and idempotence. Result semantics are
 * covered end to end by `org.apache.spark.sql.SelfJoinInequalityRewriteSuite`.
 *
 * The rule only ever fires inside an uncorrelated `IN` subquery, so every case here wraps the
 * candidate plan in one and inspects the rewritten [[ListQuery]] plan. Because `IN` consumes the
 * subquery output positionally, [[assertRewritten]] also checks that the output arity and data
 * types are preserved, that no dangling references or duplicate `ExprId`s are introduced, and that
 * the rule is idempotent -- the rule runs in a `Once` batch that is not listed in
 * `Optimizer.excludedOnceBatches`, so `RuleExecutor` re-runs it under tests and requires a fixed
 * point.
 */
class RewriteSelfJoinInequalityToAggregateSuite extends PlanTest {

  private object Optimize extends RuleExecutor[LogicalPlan] {
    val batches = Batch("Rewrite self-join inequality", Once,
      RewriteSelfJoinInequalityToAggregate) :: Nil
  }

  private val confKey = SQLConf.REWRITE_SELF_JOIN_INEQUALITY_TO_AGGREGATE_ENABLED.key

  /** The rule is disabled by default, so every case that expects it to run has to turn it on. */
  private def optimize(plan: LogicalPlan): LogicalPlan =
    withSQLConf(confKey -> "true") { Optimize.execute(plan) }

  // `ws` stands in for the TPC-DS web_sales self-join: `k` is the equality key (ws_order_number)
  // and `v` the inequality value (ws_warehouse_sk). `wr` plays web_returns, the other side of the
  // enclosing join, and `outer` supplies the `IN` context.
  private val ws = LocalRelation($"k".int, $"v".int)
  private val ws3 = LocalRelation($"a".int, $"b".int, $"v".int)
  private val dbl = LocalRelation($"dk".double, $"dv".int)
  private val wr = LocalRelation($"rk".int)
  private val outer = LocalRelation($"o".int)

  // --------------------------------------------------------------------------
  // Plan builders.
  // --------------------------------------------------------------------------

  /** `l.k = r.k AND l.v <> r.v`, the canonical shape this rule recognizes. */
  private def defaultCondition(left: Seq[Attribute], right: Seq[Attribute]): Expression =
    And(EqualTo(left.head, right.head), Not(EqualTo(left(1), right(1))))

  private def joinOn(
      left: LogicalPlan,
      right: LogicalPlan,
      hint: JoinHint = JoinHint.NONE)(
      condition: (Seq[Attribute], Seq[Attribute]) => Expression): Join =
    Join(left, right, Inner, Some(condition(left.output, right.output)), hint)

  /** An inequality self-join of `base` with a fresh instance of itself. */
  private def selfJoin(base: LocalRelation = ws): Join =
    joinOn(base, base.newInstance())(defaultCondition)

  /** The subquery Project that exposes only the left equality key, as `IN` requires. */
  private def keyProject(join: Join): Project = Project(Seq(join.left.output.head), join)

  /**
   * Wraps `sub` in `... IN (sub)`. The left-hand values are typed NULL literals so that a subquery
   * of any arity and column type stays resolved; only the plan shape matters here.
   */
  private def inSubquery(sub: LogicalPlan): LogicalPlan =
    Filter(
      InSubquery(
        sub.output.map(a => Literal(null, a.dataType)),
        ListQuery(sub, numCols = sub.output.size)),
      outer)

  /** `wr JOIN sub ON rk = sub.head`, i.e. the self-join nested under another inner join. */
  private def nest(sub: LogicalPlan, onLeft: Boolean = false): Project = {
    val join = if (onLeft) {
      Join(sub, wr, Inner, Some(EqualTo(sub.output.head, wr.output.head)), JoinHint.NONE)
    } else {
      Join(wr, sub, Inner, Some(EqualTo(wr.output.head, sub.output.head)), JoinHint.NONE)
    }
    Project(Seq(wr.output.head), join)
  }

  // --------------------------------------------------------------------------
  // Assertions.
  // --------------------------------------------------------------------------

  private def subqueryPlan(plan: LogicalPlan): LogicalPlan =
    plan.expressions.flatMap(_.collect { case lq: ListQuery => lq.plan }).head

  private def countJoins(plan: LogicalPlan): Int = plan.collect { case j: Join => j }.size

  private def countAggregates(plan: LogicalPlan): Int =
    plan.collect { case a: Aggregate => a }.size

  /**
   * Runs the rule over `IN (sub)` and returns the rewritten subquery plan, checking the invariants
   * every rewritten shape must satisfy.
   */
  private def assertRewritten(
      sub: LogicalPlan,
      expectedJoins: Int = 0,
      expectedAggregates: Int = 1): LogicalPlan = {
    val once = optimize(inSubquery(sub))
    val rewritten = subqueryPlan(once)

    assert(!rewritten.fastEquals(sub), s"expected a rewrite of\n${sub.treeString}")
    assert(countJoins(rewritten) === expectedJoins,
      s"unexpected number of surviving joins in\n${rewritten.treeString}")
    assert(countAggregates(rewritten) === expectedAggregates,
      s"unexpected number of aggregations in\n${rewritten.treeString}")

    // `IN` reads the subquery output positionally, so arity and types must survive the rewrite.
    assert(rewritten.output.map(_.dataType) === sub.output.map(_.dataType),
      s"subquery output types changed:\n${rewritten.treeString}")

    assert(rewritten.resolved, s"rewritten plan is unresolved:\n${rewritten.treeString}")
    val danglingRefs = rewritten.collect {
      case p if p.children.nonEmpty && p.missingInput.nonEmpty => p.missingInput
    }
    assert(danglingRefs.isEmpty, s"dangling references $danglingRefs in\n${rewritten.treeString}")
    val duplicateIds = LogicalPlanIntegrity.validateExprIdUniqueness(rewritten)
    assert(duplicateIds.isEmpty, s"$duplicateIds in\n${rewritten.treeString}")

    assertIdempotent(sub)
    rewritten
  }

  private def assertNotRewritten(sub: LogicalPlan): Unit = {
    val rewritten = subqueryPlan(optimize(inSubquery(sub)))
    assert(rewritten.fastEquals(sub), s"expected no rewrite, but got\n${rewritten.treeString}")
  }

  /** A `Once` batch has no fixed-point loop, so reaching one is part of the rule's contract. */
  private def assertIdempotent(sub: LogicalPlan): Unit = {
    val once = optimize(inSubquery(sub))
    comparePlans(optimize(once), once)
  }

  /** The MIN/MAX aggregation this rule builds over one copy of the relation. */
  private def expectedAggregate(
      relation: LogicalPlan,
      equiKeys: Seq[Attribute],
      neq: Attribute): LogicalPlan = {
    val min = Alias(Min(neq).toAggregateExpression(), "_self_join_min")()
    val max = Alias(Max(neq).toAggregateExpression(), "_self_join_max")()
    val filtered = Filter(equiKeys.map(IsNotNull(_): Expression).reduce(And), relation)
    Filter(
      Not(EqualTo(min.toAttribute, max.toAttribute)),
      Aggregate(equiKeys, equiKeys ++ Seq(min, max), filtered))
  }

  // --------------------------------------------------------------------------
  // Configuration.
  // --------------------------------------------------------------------------

  test("the rewrite is opt-in and fires only when the config is enabled") {
    assert(!SQLConf.get.getConf(SQLConf.REWRITE_SELF_JOIN_INEQUALITY_TO_AGGREGATE_ENABLED),
      "the rewrite must stay opt-in; flipping the default silently changes every user's plans " +
        "and every golden plan under PlanStabilitySuite")

    val sub = keyProject(selfJoin())
    Seq(false, true).foreach { enabled =>
      withSQLConf(confKey -> enabled.toString) {
        val rewritten = subqueryPlan(Optimize.execute(inSubquery(sub)))
        assert(rewritten.fastEquals(sub) === !enabled,
          s"$confKey=$enabled produced\n${rewritten.treeString}")
      }
    }
  }

  // --------------------------------------------------------------------------
  // The direct shape: Project above the self-join.
  // --------------------------------------------------------------------------

  test("direct self-join under IN is rewritten to a MIN/MAX aggregation") {
    val join = selfJoin()
    val Seq(lk, lv) = join.left.output

    // Comparing against the fully spelled out replacement pins all of: the self-join is gone, the
    // NULL equality keys are filtered out below the aggregation, the grouping key is the surviving
    // left key, and the inequality became MIN(v) <> MAX(v) evaluated after grouping.
    val expected =
      Project(Seq(Alias(lk, lk.name)()), expectedAggregate(join.left, Seq(lk), lv))

    comparePlans(assertRewritten(keyProject(join)), expected)
  }

  test("a computed projection over an equality key is rewritten") {
    // Regression test for the direct/nested capability gap: the direct path used to accept only a
    // bare Attribute, an Alias over one, or a foldable expression. Any expression whose references
    // all remap to surviving equality keys is sound, because the equi-join proves the two copies of
    // the key hold the same non-NULL value.
    Seq("left" -> 0, "right" -> 1).foreach { case (side, index) =>
      withClue(s"$side copy of the key: ") {
        val join = selfJoin()
        val key = Seq(join.left, join.right)(index).output.head
        val lk = join.left.output.head
        val projected = Alias(Add(key, Literal(1)), "k1")()

        val rewritten = assertRewritten(Project(Seq(projected), join))

        // Whichever copy was projected, the surviving expression must read the retained key.
        assert(rewritten.expressions.map(_.references).reduce(_ ++ _) === AttributeSet(Seq(lk)),
          s"the projection was not remapped to the surviving key:\n${rewritten.treeString}")
      }
    }
  }

  test("both copies of an equality key keep distinct output identities") {
    // `SELECT l.k, r.k` collapses to two reads of the same surviving key. They must not collapse to
    // the same ExprId: `IN` consumes the two slots positionally and later rules would treat one
    // ExprId produced twice as a broken plan.
    val join = selfJoin()
    val lk = join.left.output.head
    val sub = Project(Seq(lk, join.right.output.head), join)

    val rewritten = assertRewritten(sub).asInstanceOf[Project]

    assert(rewritten.projectList.size === 2, s"expected two slots:\n${rewritten.treeString}")
    assert(rewritten.output.map(_.exprId).distinct.size === 2,
      s"the two key slots must have distinct ExprIds:\n${rewritten.treeString}")
    rewritten.projectList.foreach { slot =>
      assert(slot.references === AttributeSet(Seq(lk)),
        s"slot $slot does not read the surviving key")
    }
  }

  test("multiple equality keys are all kept as grouping keys") {
    val join = joinOn(ws3, ws3.newInstance()) { (l, r) =>
      And(And(EqualTo(l(0), r(0)), EqualTo(l(1), r(1))), Not(EqualTo(l(2), r(2))))
    }
    val Seq(a, b, v) = join.left.output
    val expected = Project(
      Seq(Alias(a, a.name)(), Alias(b, b.name)()),
      expectedAggregate(join.left, Seq(a, b), v))

    comparePlans(assertRewritten(Project(Seq(a, b), join)), expected)
  }

  test("foldable projection slots are preserved") {
    // `o IN (SELECT 1 FROM ws l JOIN ws r ON ...)`: the projected constant is row-invariant, so it
    // survives the rewrite even though it is not an equality key.
    val join = selfJoin()
    val one = Alias(Literal(1), "one")()
    val rewritten = assertRewritten(Project(Seq(one), join))
    assert(rewritten.asInstanceOf[Project].projectList === Seq(one))
  }

  test("IS NOT NULL conjuncts on join keys are accepted") {
    // InferFiltersFromConstraints commonly adds these to the join condition before this rule runs.
    val join = joinOn(ws, ws.newInstance()) { (l, r) =>
      Seq(EqualTo(l.head, r.head), IsNotNull(l.head), IsNotNull(r(1)), Not(EqualTo(l(1), r(1))))
        .reduce(And)
    }
    assertRewritten(keyProject(join))
  }

  test("an IS NOT NULL already implied by the relation is not emitted twice") {
    // InferFiltersFromConstraints runs long before this batch and normally pushes IS NOT NULL on
    // the join keys down to both sides, so the filter this rule wants is usually already there.
    // Emitting it again leaves a redundant predicate that no later PruneFilters fixed point will
    // clean up, which shows up as plan noise in every golden plan.
    def filtered(relation: LocalRelation): Filter =
      Filter(IsNotNull(relation.output.head), relation)
    val join = joinOn(filtered(ws), filtered(ws.newInstance()))(defaultCondition)
    val key = join.left.output.head

    // The rule reads `child.constraints` to find out what is already implied, which is only
    // populated when constraint propagation is on.
    val rewritten = withSQLConf(SQLConf.CONSTRAINT_PROPAGATION_ENABLED.key -> "true") {
      assertRewritten(keyProject(join))
    }

    val isNotNulls = rewritten.collect { case f: Filter => f }
      .flatMap(f => splitConjunctivePredicates(f.condition))
      .collect { case p @ IsNotNull(a: Attribute) if a.exprId == key.exprId => p }
    assert(isNotNulls.size === 1,
      s"expected exactly one IsNotNull(${key.name}), got ${isNotNulls.size} in" +
        s"\n${rewritten.treeString}")
  }

  test("a floating-point inequality value is supported") {
    // Only equality keys are substituted into surviving expressions; the inequality column is never
    // exposed. `<>`, MIN and MAX all use the same Spark ordering for floats -- NaN equals NaN and
    // -0.0 equals 0.0 -- so MIN(v) <> MAX(v) agrees with `l.v <> r.v` on every group.
    val base = LocalRelation($"k".int, $"v".double)
    assertRewritten(keyProject(joinOn(base, base.newInstance())(defaultCondition)))
  }

  test("a relation that declares itself repeatable is rewritten") {
    // Every other positive case here reads a LocalRelation, which is repeatable because it carries
    // its rows in the plan. The relations this rule exists for -- the file-backed tables of TPC-DS
    // q95 -- are LogicalRelations instead, and reach the rewrite only by declaring repeatability
    // through RepeatableLeafNode, so that path needs its own case.
    val left = TestRepeatableRelation(ws.output.map(_.newInstance()), isOutputRepeatable = true)
    val right = TestRepeatableRelation(left.output.map(_.newInstance()), isOutputRepeatable = true)
    assertRewritten(keyProject(joinOn(left, right)(defaultCondition)))
  }

  test("the direct rewrite is idempotent") {
    assertIdempotent(keyProject(selfJoin()))
  }

  // --------------------------------------------------------------------------
  // The enclosing-join shape: the self-join is one child of another inner join.
  // --------------------------------------------------------------------------

  test("self-join nested under an inner join is rewritten (TPC-DS q95 shape)") {
    // The second IN subquery of q95:
    //   SELECT wr_order_number FROM web_returns wr
    //   JOIN (SELECT ws1.ws_order_number FROM web_sales ws1 JOIN web_sales ws2
    //         ON ws1.ws_order_number = ws2.ws_order_number
    //        AND ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk) t
    //   ON wr.wr_order_number = t.ws_order_number
    // The top Project reads only the other side, and the enclosing condition reads only the
    // surviving equality key, which is exactly what makes the child replaceable.
    val join = selfJoin()
    val Seq(lk, lv) = join.left.output
    val Seq(rkOuter) = wr.output

    val aggregate = expectedAggregate(join.left, Seq(lk), lv)
    val newKey = Alias(lk, lk.name)()
    val expected = Project(
      Seq(rkOuter),
      Join(wr, Project(Seq(newKey), aggregate), Inner,
        Some(EqualTo(rkOuter, newKey.toAttribute)), JoinHint.NONE))

    comparePlans(assertRewritten(nest(keyProject(join)), expectedJoins = 1), expected)
  }

  test("the nested self-join is rewritten on either side of the enclosing join") {
    Seq(true, false).foreach { onLeft =>
      withClue(s"self-join on the ${if (onLeft) "left" else "right"}: ") {
        assertRewritten(nest(keyProject(selfJoin()), onLeft = onLeft), expectedJoins = 1)
      }
    }
  }

  test("both children of the enclosing join are rewritten in one pass") {
    // The rule folds over both sides within a single Once batch, so an eligible candidate must not
    // be left behind just because the other side was handled first.
    val left = keyProject(selfJoin())
    val right = keyProject(selfJoin(ws.newInstance()))
    val join = Join(left, right, Inner,
      Some(EqualTo(left.output.head, right.output.head)), JoinHint.NONE)

    val rewritten = assertRewritten(
      Project(Seq(left.output.head), join), expectedJoins = 1, expectedAggregates = 2)

    // The enclosing join must still correlate the two sides: one reference from each child.
    val surviving = rewritten.collectFirst { case j: Join => j }.get
    val refs = surviving.condition.get.references
    assert(refs.count(surviving.left.outputSet.contains) === 1 &&
      refs.count(surviving.right.outputSet.contains) === 1,
      s"the enclosing join condition degenerated: ${surviving.condition.get}")
  }

  test("a nested self-join with no wrapper Project remaps every surviving reference") {
    // The riskiest enclosing shape: with no Project between the two joins, both the enclosing join
    // condition and the top-level Project may read either copy of the key directly.
    val join = selfJoin()
    val lk = join.left.output.head
    val rk = join.right.output.head
    val Seq(rkOuter) = wr.output
    val outerJoin = Join(wr, join, Inner, Some(EqualTo(rkOuter, rk)), JoinHint.NONE)
    val sub = Project(Seq(rkOuter, lk, rk), outerJoin)

    val rewritten = assertRewritten(sub, expectedJoins = 1)

    // Every reference in the surviving plan must be produced by the operator below it, and the two
    // key slots must stay distinct: this is what the ExprId remapping exists for.
    assert(rewritten.output.map(_.exprId).distinct.size === 3,
      s"output identities collapsed:\n${rewritten.treeString}")
    val surviving = rewritten.collectFirst { case j: Join => j }.get
    assert(surviving.condition.get.references.subsetOf(
      surviving.left.outputSet ++ surviving.right.outputSet),
      s"the enclosing join condition references a removed attribute:\n${rewritten.treeString}")
  }

  test("an equality key exposed from the eliminated side is remapped to the surviving key") {
    val join = selfJoin()
    // The wrapper exposes the *right* copy's key, which does not survive the aggregation.
    val wrapper = Project(Seq(join.right.output.head), join)
    val rewritten = assertRewritten(nest(wrapper), expectedJoins = 1)

    val aggregate = rewritten.collectFirst { case a: Aggregate => a }.get
    assert(aggregate.groupingExpressions === Seq(join.left.output.head),
      s"the surviving key must come from the retained side:\n${rewritten.treeString}")
  }

  test("no rewrite when the enclosing join reads a column the aggregation removes") {
    // Only the equality keys survive grouping. If the enclosing join correlates on the inequality
    // column -- or on any other column of the eliminated self-join -- there is nothing to remap it
    // to, so the candidate must be rejected rather than silently re-pointed at a key.
    val join = selfJoin()
    val Seq(rkOuter) = wr.output

    Seq("left inequality column" -> join.left.output(1),
      "right inequality column" -> join.right.output(1)).foreach { case (name, column) =>
      withClue(s"$name: ") {
        val outerJoin = Join(wr, join, Inner, Some(EqualTo(rkOuter, column)), JoinHint.NONE)
        assertNotRewritten(Project(Seq(rkOuter), outerJoin))
      }
    }
  }

  test("the nested rewrite is idempotent") {
    // Kept separate from the direct case: the nested path folds over both children, so a
    // non-idempotent step here would not show up in the direct test.
    Seq(
      "wrapper Project, self-join on the right" -> nest(keyProject(selfJoin())),
      "wrapper Project, self-join on the left" -> nest(keyProject(selfJoin()), onLeft = true),
      "no wrapper Project" -> nest(selfJoin()),
      "both children" -> {
        val left = keyProject(selfJoin())
        val right = keyProject(selfJoin(ws.newInstance()))
        Project(Seq(left.output.head), Join(left, right, Inner,
          Some(EqualTo(left.output.head, right.output.head)), JoinHint.NONE))
      }
    ).foreach { case (name, sub) =>
      withClue(s"$name: ") {
        assertIdempotent(sub)
      }
    }
  }

  // --------------------------------------------------------------------------
  // Unsupported shapes. The rule fails closed, so each case must be left alone.
  // --------------------------------------------------------------------------

  test("no rewrite when a project expression carries a subquery") {
    // A SubqueryExpression under-reports what it depends on: `references` covers only its outer
    // attributes, and `transformUp` does not descend into its plan, so remapping the outer
    // attributes leaves the inner plan pointing at an ExprId that no longer exists. Rule ordering
    // currently rewrites correlated scalar subqueries long before this batch, so this shape is not
    // reachable through the full Optimizer today -- which is exactly why the rule is driven
    // directly here. The guard must not depend on another rule running first.
    val join = selfJoin()
    val rk = join.right.output.head
    val correlated = ScalarSubquery(
      plan = Aggregate(Nil, Seq(Alias(Max(wr.output.head).toAggregateExpression(), "m")()),
        Filter(EqualTo(wr.output.head, OuterReference(rk)), wr)),
      outerAttrs = Seq(rk))
    val sub = Project(Seq(join.left.output.head, Alias(correlated, "x")()), join)

    val plan = inSubquery(sub)
    val rewritten = withSQLConf(confKey -> "true") {
      RewriteSelfJoinInequalityToAggregate(plan)
    }
    assert(rewritten.fastEquals(plan),
      s"expected no rewrite, but got\n${subqueryPlan(rewritten).treeString}")
  }

  test("no rewrite for a correlated IN subquery") {
    // A correlated subquery may be evaluated per outer row, and de-correlation happens later, so
    // the multiplicity argument this rewrite relies on does not hold yet.
    val join = selfJoin()
    val sub = keyProject(join)
    val correlated = Filter(
      InSubquery(
        Seq(outer.output.head),
        ListQuery(sub, outerAttrs = Seq(outer.output.head), numCols = 1)),
      outer)
    val rewritten = subqueryPlan(optimize(correlated))
    assert(rewritten.fastEquals(sub), s"expected no rewrite, but got\n${rewritten.treeString}")
  }

  test("no rewrite without a Project above the self-join") {
    // A bare join exposes both sides' full schemas; replacing it would change the positional
    // output that `IN` consumes.
    assertNotRewritten(selfJoin())
  }

  test("no rewrite for a non-inner join") {
    val join = selfJoin().copy(joinType = LeftOuter)
    assertNotRewritten(keyProject(join))
  }

  test("no rewrite when the join condition is not equalities plus one inequality") {
    def reject(name: String, base: LocalRelation = ws)(
        condition: (Seq[Attribute], Seq[Attribute]) => Expression): Unit = {
      withClue(s"$name: ") {
        assertNotRewritten(keyProject(joinOn(base, base.newInstance())(condition)))
      }
    }

    // EqualNullSafe is not EqualTo: it matches NULL keys, which the rewrite's IS NOT NULL filter
    // removes but GROUP BY would keep.
    reject("EqualNullSafe key")((l, r) =>
      And(EqualNullSafe(l.head, r.head), Not(EqualTo(l(1), r(1)))))
    // A disjunction is not a conjunction of provable pairs.
    reject("disjunction")((l, r) => Or(EqualTo(l.head, r.head), Not(EqualTo(l(1), r(1)))))
    // Computed keys are not proven to correspond.
    reject("computed key")((l, r) =>
      And(EqualTo(Add(l.head, Literal(1)), r.head), Not(EqualTo(l(1), r(1)))))
    reject("computed inequality")((l, r) =>
      And(EqualTo(l.head, r.head), Not(EqualTo(Add(l(1), Literal(1)), r(1)))))
    // A residual predicate would have to be re-applied below the aggregation.
    reject("residual predicate")((l, r) =>
      And(And(EqualTo(l.head, r.head), Not(EqualTo(l(1), r(1)))), GreaterThan(l(1), Literal(0))))
    // MIN(v) <> MAX(v) proves that exactly one inequality holds somewhere in the group; it says
    // nothing about a second, independent inequality.
    reject("two inequalities", ws3)((l, r) =>
      Seq(EqualTo(l(0), r(0)), Not(EqualTo(l(1), r(1))), Not(EqualTo(l(2), r(2)))).reduce(And))
    reject("no inequality")((l, r) => EqualTo(l.head, r.head))
    reject("no equality")((l, r) => Not(EqualTo(l(1), r(1))))
    // `l.k = r.k AND l.k <> r.k` is unsatisfiable; grouping on `k` would report it as satisfied.
    reject("inequality on an equality key")((l, r) =>
      And(EqualTo(l.head, r.head), Not(EqualTo(l.head, r.head))))
    // Columns at different output positions are different columns, whatever their names.
    reject("non-corresponding columns")((l, r) =>
      And(EqualTo(l.head, r(1)), Not(EqualTo(l(1), r(1)))))
    // Both operands of a pair must come from opposite sides.
    reject("one-sided pair")((l, r) =>
      And(EqualTo(l.head, r.head), Not(EqualTo(l.head, l(1)))))
  }

  test("no rewrite when a join hint is present") {
    val hinted = joinOn(ws, ws.newInstance(),
      JoinHint(Some(HintInfo(strategy = Some(BROADCAST))), None))(defaultCondition)
    assertNotRewritten(keyProject(hinted))
  }

  test("no rewrite when the relation is not repeatable") {
    // Folding two evaluations of the relation into one is only sound if both evaluations are
    // guaranteed to see the same rows.
    def rejectRelation(name: String)(relation: LocalRelation => LogicalPlan): Unit = {
      withClue(s"$name: ") {
        assertNotRewritten(keyProject(
          joinOn(relation(ws), relation(ws.newInstance()))(defaultCondition)))
      }
    }

    rejectRelation("LIMIT")(Limit(Literal(10), _))
    rejectRelation("Sample")(Sample(0.0d, 0.5d, withReplacement = false, seed = Some(1L), _))
    // A subquery inside the relation would go from two evaluations to one.
    rejectRelation("embedded subquery")(base =>
      Filter(InSubquery(Seq(base.output.head), ListQuery(wr, numCols = 1)), base))
    rejectRelation("nondeterminism")(Filter(GreaterThan(Rand(1), Literal(0.5)), _))

    // A leaf that says nothing about repeatability. It has no expressions at all, so it is
    // trivially `deterministic`, which is exactly why `deterministic` cannot be the test: the rows
    // behind it may differ between two reads. `LogicalRDD` in sql/core is this case.
    def leaves(build: Seq[Attribute] => LogicalPlan): LogicalPlan = {
      val left = build(ws.output.map(_.newInstance()))
      keyProject(joinOn(left, build(left.output.map(_.newInstance())))(defaultCondition))
    }
    withClue("undeclared leaf: ") {
      assertNotRewritten(leaves(TestLeafRelation(_)))
    }
    // A relation that declares itself non-repeatable, as a LogicalRelation over an untrusted file
    // format or a non-file BaseRelation does.
    withClue("leaf declaring itself non-repeatable: ") {
      assertNotRewritten(leaves(TestRepeatableRelation(_, isOutputRepeatable = false)))
    }

    // A CTE reference is a leaf whose definition is not visible here; it may hide a LIMIT, a
    // Sample, nondeterminism or recursion.
    val cte = CTERelationRef(
      cteId = 1,
      _resolved = true,
      output = Seq(AttributeReference("k", IntegerType)(), AttributeReference("v", IntegerType)()),
      isStreaming = false)
    assertNotRewritten(keyProject(joinOn(cte, cte.newInstance())(defaultCondition)))
  }

  test("no rewrite when the relation carries an expression outside the trust boundary") {
    // `deterministic` is not a repeatability proof for expressions either, which is why the Rand
    // case above only covers the easy half of this guard. A runtime-replaceable expression such as
    // `aes_encrypt` becomes a deterministic-looking StaticInvoke during optimization while still
    // drawing a fresh random initialization vector on every evaluation, and an opaque or
    // user-defined expression can do anything at all while declaring itself deterministic. Two
    // evaluations of such a relation do not agree, so they must not be folded into one.
    val base = LocalRelation($"k".int, $"v".int, $"b".binary)

    def candidate(build: Seq[Attribute] => Expression): Project = {
      def relation(source: LocalRelation): LogicalPlan =
        Project(source.output :+ Alias(build(source.output), "x")(), source)

      keyProject(joinOn(relation(base), relation(base.newInstance()))(defaultCondition))
    }

    // Control: the same shape carrying an ordinary Catalyst expression is rewritten, so the cases
    // below are rejected for the expression they hold and not for the shape holding it.
    withClue("Catalyst expression: ") {
      assertRewritten(candidate(out => Add(out(1), Literal(1))))
    }

    withClue("aes_encrypt, as ReplaceExpressions leaves it: ") {
      // No initialization vector is supplied, which is the case where `aes_encrypt` generates a
      // fresh random one per evaluation -- the exact expression this guard exists for. The empty
      // binary literals stand in for the defaults `new AesEncrypt(input, key)` would supply as
      // strings: this rule runs without the analyzer, so nothing would cast them to binary.
      val key = Literal("1234567890abcdef".getBytes(StandardCharsets.UTF_8))
      val noIv = Literal(Array.empty[Byte])
      def aes(input: Expression): AesEncrypt =
        AesEncrypt(input, key, Literal("GCM"), Literal("DEFAULT"), noIv, noIv)

      // Pin down what makes this case interesting, so that changing the arguments cannot quietly
      // turn it into a repeatable expression: the replacement reports itself deterministic, yet two
      // evaluations of the same input disagree.
      val probe = aes(Literal("plaintext".getBytes(StandardCharsets.UTF_8))).replacement
      assert(probe.deterministic)
      assert(probe.eval(null).asInstanceOf[Array[Byte]].toSeq !==
        probe.eval(null).asInstanceOf[Array[Byte]].toSeq)

      assertNotRewritten(candidate(out => aes(out(2)).replacement))
    }
    withClue("user-defined expression: ") {
      // One function value shared by both copies, so that the two relations stay canonically equal
      // and the candidate is rejected for the expression it holds rather than for its shape.
      val udf: Any => Any = v => v
      assertNotRewritten(candidate(out => ScalaUDF(udf, IntegerType, Seq(out(1)))))
    }
    withClue("expression outside the Catalyst expression namespace: ") {
      assertNotRewritten(candidate(out => TestOpaqueExpression(out(1))))
    }
  }

  test("no rewrite for non-binary-equality collations") {
    // Under a collation such as UTF8_LCASE, 'a' and 'A' are equal, so `l.v <> r.v` does not agree
    // with the binary comparison MIN/MAX would perform.
    def collated(name: String): LocalRelation = LocalRelation(
      AttributeReference("k", if (name == "k") StringType("UTF8_LCASE") else IntegerType)(),
      AttributeReference("v", if (name == "v") StringType("UTF8_LCASE") else IntegerType)())

    Seq("k", "v").foreach { column =>
      withClue(s"collated $column: ") {
        val base = collated(column)
        assertNotRewritten(keyProject(joinOn(base, base.newInstance())(defaultCondition)))
      }
    }
  }

  test("no rewrite when an equality key contains floating point") {
    // Floating-point equality is defined on normalized values: -0.0 equals 0.0 and every NaN
    // compares equal. Two rows can therefore join on keys that are equal but observably different,
    // while the aggregation emits a single representative per group. Any surviving projection that
    // can tell those apart -- a CAST to string, say -- would read a different value after the
    // rewrite, so a floating-point equality key is rejected outright instead of being reasoned
    // about per projection.
    Seq(
      "double key" -> dbl,
      "float key" -> LocalRelation($"k".float, $"v".int),
      "struct holding a double" -> LocalRelation(
        AttributeReference("k", StructType(Seq(StructField("d", DoubleType))))(),
        AttributeReference("v", IntegerType)())
    ).foreach { case (name, base) =>
      withClue(s"$name: ") {
        assertNotRewritten(keyProject(joinOn(base, base.newInstance())(defaultCondition)))
      }
    }

    // The normalization wrapper Spark inserts for floating-point join keys is not unwrapped
    // either. FinishAnalysis runs NormalizeFloatingNumbers before this batch (the optimizer runs it
    // a second time after RewriteSubquery, for the joins that batch creates), so the wrapper can
    // reach this rule; `extractJoinAttribute` accepts a bare Attribute only, which fails closed.
    withClue("normalized double key: ") {
      assertNotRewritten(keyProject(joinOn(dbl, dbl.newInstance()) { (l, r) =>
        And(
          EqualTo(
            KnownFloatingPointNormalized(NormalizeNaNAndZero(l.head)),
            KnownFloatingPointNormalized(NormalizeNaNAndZero(r.head))),
          Not(EqualTo(l(1), r(1))))
      }))
    }
  }

  test("no rewrite when the inequality value type is not orderable") {
    // MapType is binary-stable but has no ordering, so MIN/MAX cannot express the inequality.
    // The analyzer rejects map comparison, so this branch is defensive only.
    val base = LocalRelation(
      AttributeReference("k", IntegerType)(),
      AttributeReference("v", MapType(IntegerType, IntegerType))())
    assertNotRewritten(keyProject(joinOn(base, base.newInstance())(defaultCondition)))
  }

  test("no rewrite when the surviving Project exposes the inequality column") {
    val join = selfJoin()
    // The inequality column cannot survive the aggregation: MIN/MAX collapses each group to one
    // row, and any individual `v` of that group is gone.
    assertNotRewritten(Project(join.left.output, join))
    assertNotRewritten(nest(Project(join.left.output, join)))
  }

  test("no rewrite for shapes above the candidate that the matcher does not enumerate") {
    val join = selfJoin()
    val key = join.left.output.head

    // Documented limitation: only `Project(Join)` and `Project(Join(Project(Join), _))` are
    // matched. PushPredicateThroughJoin absorbs Inner cross-side conjuncts into the join
    // condition, so a surviving Filter here is rare in practice.
    withClue("Filter between the Project and the self-join: ") {
      assertNotRewritten(Project(Seq(key), Filter(IsNotNull(key), join)))
    }

    // Documented limitation: `SELECT DISTINCT` becomes an Aggregate wrapper, which the matcher does
    // not look through even though DISTINCT already makes multiplicity unobservable.
    withClue("Distinct wrapper: ") {
      val distinct = Aggregate(Seq(key), Seq(key), keyProject(join))
      assertNotRewritten(distinct)
    }

    // Documented limitation: the enclosing-join shape is enumerated for exactly one level.
    withClue("two levels of nesting: ") {
      val inner = nest(keyProject(selfJoin()))
      val wr2 = wr.newInstance()
      val outerJoin = Join(wr2, inner, Inner,
        Some(EqualTo(wr2.output.head, inner.output.head)), JoinHint.NONE)
      assertNotRewritten(Project(Seq(wr2.output.head), outerJoin))
    }
  }

  test("no rewrite when an expression above the candidate is outside the trust boundary") {
    // The relation itself is only half of the trust boundary. Collapsing the self-join's
    // multiplicity also means every expression above it runs once per group where it used to run
    // once per matching pair, so an expression that returns a fresh value per evaluation emits a
    // different value set than the self-join did, and `IN` compares against that set. Each
    // placement below is paired with a control carrying an ordinary Catalyst expression, so a
    // rejection is attributable to the expression rather than to the shape holding it.
    val direct: (Attribute => Expression) => Project = build => {
      val join = selfJoin()
      Project(Seq(Alias(build(join.left.output.head), "x")()), join)
    }

    val nestedTopProject: (Attribute => Expression) => Project = build => {
      val sub = keyProject(selfJoin())
      val join =
        Join(wr, sub, Inner, Some(EqualTo(wr.output.head, sub.output.head)), JoinHint.NONE)
      Project(Seq(Alias(build(wr.output.head), "x")()), join)
    }

    val enclosingCondition: (Attribute => Expression) => Project = build => {
      val sub = keyProject(selfJoin())
      val join = Join(wr, sub, Inner,
        Some(EqualTo(build(wr.output.head), build(sub.output.head))), JoinHint.NONE)
      Project(Seq(wr.output.head), join)
    }

    // One function value shared by every candidate, so that relations stay canonically equal.
    val udf: Any => Any = v => v

    Seq(
      ("the top Project of the direct shape", direct, 0),
      ("the top Project of the nested shape", nestedTopProject, 1),
      ("an enclosing join condition", enclosingCondition, 1)).foreach {
      case (where, candidate, survivingJoins) =>
        withClue(s"Catalyst expression in $where: ") {
          assertRewritten(candidate(a => Add(a, Literal(1))), expectedJoins = survivingJoins)
        }
        withClue(s"opaque expression in $where: ") {
          assertNotRewritten(candidate(TestOpaqueExpression(_)))
        }
        withClue(s"user-defined expression in $where: ") {
          assertNotRewritten(candidate(a => ScalaUDF(udf, IntegerType, Seq(a))))
        }
    }
  }

}

/**
 * A leaf relation that says nothing about repeatability, like an RDD- or connector-backed one.
 * Canonicalization normalizes the output so that two instances differing only in `ExprId`s compare
 * equal, as [[LocalRelation]] and `LogicalRelation` do.
 */
private case class TestLeafRelation(output: Seq[Attribute]) extends LeafNode {
  override def doCanonicalize(): LogicalPlan =
    copy(output = output.map(QueryPlan.normalizeExpressions(_, output)))
}

/** A leaf relation that declares whether two reads return the same rows, like a file source. */
private case class TestRepeatableRelation(output: Seq[Attribute], isOutputRepeatable: Boolean)
  extends RepeatableLeafNode {
  override def doCanonicalize(): LogicalPlan =
    copy(output = output.map(QueryPlan.normalizeExpressions(_, output)))
}

/**
 * A deterministic expression that lives outside the Catalyst expression namespace, like an
 * expression contributed by a data source or another Spark module.
 */
private case class TestOpaqueExpression(child: Expression)
  extends UnaryExpression with CodegenFallback {
  override def dataType: DataType = child.dataType
  override def eval(input: InternalRow): Any = child.eval(input)
  override protected def withNewChildInternal(newChild: Expression): Expression =
    copy(child = newChild)
}
