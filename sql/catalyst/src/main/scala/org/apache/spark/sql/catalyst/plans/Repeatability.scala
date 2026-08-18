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
package org.apache.spark.sql.catalyst.plans

import org.apache.spark.sql.catalyst.expressions.{AesEncrypt, NonSQLExpression, UserDefinedExpression}
import org.apache.spark.sql.catalyst.trees.TreePattern.CURRENT_LIKE

/**
 * Shared checks for whether evaluating a plan more than once returns the same rows.
 *
 * This is only about the expressions a plan carries. A caller also has to establish that the plan's
 * operators and leaves are repeatable, which is source specific: see
 * [[org.apache.spark.sql.catalyst.plans.logical.RepeatableLeafNode]] for the leaf contract.
 */
private[sql] object Repeatability {

  private val catalystExpressionPackage = "org.apache.spark.sql.catalyst.expressions."

  /**
   * Whether every expression of `plan` honors `Expression.deterministic`'s repeatability contract.
   *
   * `deterministic` is declared per expression and is not enough on its own. A runtime-replaceable
   * expression such as `aes_encrypt` becomes a deterministic-looking `StaticInvoke` during
   * optimization while still drawing a fresh random initialization vector on every evaluation, and
   * an opaque or user-defined expression can do anything at all while declaring itself
   * deterministic. Treat the Catalyst expression namespace as the trust boundary: expressions
   * outside it fail closed, and `AesEncrypt`, opaque and user-defined expressions are rejected
   * explicitly.
   */
  def hasSafeExpressions(plan: QueryPlan[_]): Boolean = {
    plan.expressions.forall { expression =>
      !expression.exists {
        case _: AesEncrypt | _: NonSQLExpression | _: UserDefinedExpression => true
        case value => !value.deterministic || value.containsPattern(CURRENT_LIKE) ||
          !value.getClass.getName.startsWith(catalystExpressionPackage)
      }
    }
  }
}
