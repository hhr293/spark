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
package org.apache.spark.sql

import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Join}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

/**
 * End-to-end semantic tests for the `RewriteSelfJoinInequalityToAggregate` optimizer rule. The rule
 * replaces `l.k = r.k AND l.v <> r.v` with `GROUP BY k HAVING MIN(v) <> MAX(v)`, which is only
 * equivalent because of how NULL keys, NULL values, duplicate rows and floating-point equality
 * behave. Each case here therefore runs the same query with the rule disabled and enabled and
 * requires both to produce the same answer; the datasets are built so that a plausible mistake in
 * the rewrite changes that answer.
 *
 * Matcher, fail-closed and plan-shape coverage lives in
 * `org.apache.spark.sql.catalyst.optimizer.RewriteSelfJoinInequalityToAggregateSuite`.
 */
class SelfJoinInequalityRewriteSuite extends QueryTest with SharedSparkSession {

  private val confKey = SQLConf.REWRITE_SELF_JOIN_INEQUALITY_TO_AGGREGATE_ENABLED.key

  private def withViews(definitions: (String, String)*)(f: => Unit): Unit = {
    withTempView(definitions.map(_._1): _*) {
      definitions.foreach { case (name, definition) =>
        sql(s"CREATE OR REPLACE TEMP VIEW $name AS $definition")
      }
      f
    }
  }

  private def values(columns: String, rows: String): String =
    s"SELECT * FROM VALUES $rows AS t($columns)"

  /**
   * Runs `query` with the rule disabled, pins that answer against `expected`, then runs it with the
   * rule enabled and requires the answer the disabled plan produced -- so the expectation is
   * Spark's own self-join semantics, not a hand-derived one. The join counts guard against a
   * vacuous pass: without them a rule that stopped firing would still be green.
   */
  private def checkRewrite(
      query: String,
      expected: Seq[Row],
      joinsWhenDisabled: Int,
      joinsWhenEnabled: Int): Unit = {
    val disabled = withSQLConf(confKey -> "false") {
      val df = sql(query)
      withClue(s"$confKey=false, optimized plan:\n${df.queryExecution.optimizedPlan}\n") {
        checkAnswer(df, expected)
        assert(countJoins(df) === joinsWhenDisabled)
      }
      df.collect().toSeq
    }
    withSQLConf(confKey -> "true") {
      val df = sql(query)
      withClue(s"$confKey=true, optimized plan:\n${df.queryExecution.optimizedPlan}\n") {
        checkAnswer(df, disabled)
        assert(countJoins(df) === joinsWhenEnabled)
        // The rewrite is the only thing in these queries that can introduce an aggregation.
        assert(df.queryExecution.optimizedPlan.exists(_.isInstanceOf[Aggregate]) ===
          (joinsWhenEnabled < joinsWhenDisabled))
      }
    }
  }

  private def countJoins(df: DataFrame): Int =
    df.queryExecution.optimizedPlan.collect { case j: Join => j }.size

  private val directQuery =
    """SELECT id FROM ids WHERE id IN (
      |  SELECT l.k FROM ws l JOIN ws r ON l.k = r.k AND l.v <> r.v)""".stripMargin

  test("IN: NULL keys, NULL values, duplicates and self-pairs") {
    // One dataset covering every way a group can fail to hold two distinct values:
    //   k = 1    -> 10 and 20, two distinct values                             -> returned
    //   k = 2    -> 30 twice; the rewrite must not report a group of duplicates -> not returned
    //   k = 3    -> NULL and 40; `NULL <> 40` is NULL, and so is MIN/MAX of a
    //               group whose only other value is NULL                       -> not returned
    //   k = NULL -> `l.k = r.k` never matches, so GROUP BY must not form this
    //               group even though it holds two distinct values             -> not returned
    // The self-pair (every row joined to itself) is excluded by `l.v <> r.v` in the join and by
    // MIN <> MAX in the rewrite, which is why k = 2 is the interesting row here.
    withViews(
      "ws" -> values("k, v",
        "(1, 10), (1, 20), (2, 30), (2, 30), (3, NULL), (3, 40), (NULL, 10), (NULL, 20)"),
      "ids" -> values("id", "(1), (2), (3), (NULL)")) {
      checkRewrite(directQuery, Seq(Row(1)), joinsWhenDisabled = 2, joinsWhenEnabled = 1)
    }
  }

  test("NOT IN: a NULL key must never leak into the subquery output") {
    // The sharpest test of the IS NOT NULL filter the rewrite inserts. `l.k = r.k` cannot match a
    // NULL key, so the self-join never emits one; GROUP BY k, by contrast, happily forms a NULL
    // group, and here that group does hold two distinct values (10 and 20). Under NOT IN a single
    // NULL in the subquery makes the null-aware anti join reject every outer row, so dropping the
    // filter turns `(2, 3)` into an empty result. That makes IS NOT NULL a correctness
    // requirement, not plan tidiness.
    val query =
      """SELECT id FROM ids WHERE id NOT IN (
        |  SELECT l.k FROM ws l JOIN ws r ON l.k = r.k AND l.v <> r.v)""".stripMargin
    withViews(
      "ws" -> values("k, v", "(1, 10), (1, 20), (NULL, 10), (NULL, 20)"),
      "ids" -> values("id", "(1), (2), (3)")) {
      checkRewrite(query, Seq(Row(2), Row(3)), joinsWhenDisabled = 2, joinsWhenEnabled = 1)
    }
  }

  test("floating-point inequality values: 0.0 vs -0.0 and NaN") {
    // Spark SQL equality treats 0.0 = -0.0 and NaN = NaN as true, and MIN/MAX order NaN as the
    // largest value, so `MIN(v) <> MAX(v)` has to agree with `l.v <> r.v` on all of:
    //   k = 1 -> 0.0 and -0.0 are the same value  -> not returned
    //   k = 2 -> two NaNs are the same value      -> not returned
    //   k = 3 -> NaN and 1.0 are distinct         -> returned
    //   k = 4 -> NULL never satisfies `<>`        -> not returned
    withViews(
      "ws" -> values("k, v",
        """(1, 0.0D), (1, CAST('-0.0' AS DOUBLE)),
          |(2, CAST('NaN' AS DOUBLE)), (2, CAST('NaN' AS DOUBLE)),
          |(3, CAST('NaN' AS DOUBLE)), (3, 1.0D),
          |(4, CAST(NULL AS DOUBLE)), (4, 1.0D)""".stripMargin),
      "ids" -> values("id", "(1), (2), (3), (4)")) {
      checkRewrite(directQuery, Seq(Row(3)), joinsWhenDisabled = 2, joinsWhenEnabled = 1)
    }
  }

  test("a floating-point equality key is not rewritten") {
    // -0.0 and 0.0 are equal keys but distinguishable values. The self-join emits the key of every
    // matching row -- here both '0.0' and '-0.0' -- whereas an aggregation emits one normalized
    // representative per group, which a projection that can tell them apart, such as this CAST to
    // string, would observe. The rule must therefore decline the candidate outright, so the join
    // count is required to stay the same with the rule enabled.
    val query =
      """SELECT s FROM strs WHERE s IN (
        |  SELECT CAST(r.k AS STRING) FROM ws l JOIN ws r ON l.k = r.k AND l.v <> r.v)"""
        .stripMargin
    withViews(
      "ws" -> values("k, v", "(0.0D, 1), (CAST('-0.0' AS DOUBLE), 2)"),
      "strs" -> values("s", "('0.0'), ('-0.0')")) {
      checkRewrite(query, Seq(Row("0.0"), Row("-0.0")),
        joinsWhenDisabled = 2, joinsWhenEnabled = 2)
    }
  }

  test("multiple equality keys and a multi-column IN") {
    // `IN` consumes the subquery output positionally, so a rewrite that reordered or dropped a
    // grouping key would compare the wrong columns. A NULL in either key must disqualify the
    // group, not just a NULL in the first one.
    val query =
      """SELECT k1, k2 FROM keys WHERE (k1, k2) IN (
        |  SELECT l.k1, l.k2 FROM ws l JOIN ws r
        |    ON l.k1 = r.k1 AND l.k2 = r.k2 AND l.v <> r.v)""".stripMargin
    withViews(
      "ws" -> values("k1, k2, v",
        """(1, 1, 10), (1, 1, 20), (2, 2, 30), (2, 2, 30),
          |(NULL, 1, 10), (NULL, 1, 20), (3, NULL, 10), (3, NULL, 20)""".stripMargin),
      "keys" -> values("k1, k2", "(1, 1), (1, 2), (2, 2), (NULL, 1), (3, NULL)")) {
      checkRewrite(query, Seq(Row(1, 1)), joinsWhenDisabled = 2, joinsWhenEnabled = 1)
    }
  }

  test("self-join nested under another inner join (TPC-DS q95 shape)") {
    // The shape the rule has to handle for q95: the self-join is one child of an enclosing join,
    // whose condition and whose parent projection both have to keep working after the child is
    // replaced by an aggregation.
    val query =
      """SELECT id FROM ids WHERE id IN (
        |  SELECT wr.rk FROM wr JOIN (
        |    SELECT l.k FROM ws l JOIN ws r ON l.k = r.k AND l.v <> r.v) t
        |  ON wr.rk = t.k)""".stripMargin
    withViews(
      "ws" -> values("k, v", "(1, 10), (1, 20), (2, 30), (2, 30), (NULL, 10), (NULL, 20)"),
      "wr" -> values("rk", "(1), (2), (3), (NULL)"),
      "ids" -> values("id", "(1), (2), (3)")) {
      checkRewrite(query, Seq(Row(1)), joinsWhenDisabled = 3, joinsWhenEnabled = 2)
    }
  }

  // A leaf relation is rejected unless it declares that reading it twice returns the same rows,
  // which is what makes replacing two evaluations by one sound. The two cases below are the ones
  // that decide whether this rule does anything at all and whether it stays sound: every other test
  // here reads a temp view over VALUES, that is a LocalRelation carrying its rows in the plan.

  test("a file-backed table is repeatable and is rewritten") {
    // The reason the rule exists: the TPC-DS q95 tables are files, and a LogicalRelation over a
    // trusted file format is the only leaf outside Catalyst's own pure ones that qualifies. If this
    // case stops firing, the rule no longer speeds up q95.
    withTable("ws") {
      sql("CREATE TABLE ws (k INT, v INT) USING parquet")
      sql("INSERT INTO ws VALUES (1, 10), (1, 20), (2, 30), (2, 30)")
      withViews("ids" -> values("id", "(1), (2), (3)")) {
        checkRewrite(directQuery, Seq(Row(1)), joinsWhenDisabled = 2, joinsWhenEnabled = 1)
      }
    }
  }

  test("a file-backed relation read best-effort is not repeatable and is not rewritten") {
    // A best-effort read may silently skip a file that is missing or corrupt, so two reads of the
    // same relation can return different rows and the rewrite would answer a question about data
    // the self-join never saw. The option is what makes the difference here: the same table is
    // rewritten in the test above.
    withTable("ws") {
      sql("CREATE TABLE ws (k INT, v INT) USING parquet")
      sql("INSERT INTO ws VALUES (1, 10), (1, 20), (2, 30), (2, 30)")
      withViews("ids" -> values("id", "(1), (2), (3)")) {
        Seq(SQLConf.IGNORE_MISSING_FILES.key, SQLConf.IGNORE_CORRUPT_FILES.key).foreach { key =>
          withClue(s"$key=true: ") {
            withSQLConf(key -> "true") {
              checkRewrite(directQuery, Seq(Row(1)), joinsWhenDisabled = 2, joinsWhenEnabled = 2)
            }
          }
        }
      }
    }
  }

  test("a relation option overrides a strict session config") {
    // FileSourceOptions resolves each option against the session config, so a relation created with
    // the best-effort option is not repeatable even while the session asks for strict reads.
    withTempPath { path =>
      spark.sql(values("k, v", "(1, 10), (1, 20), (2, 30), (2, 30)"))
        .write.parquet(path.getCanonicalPath)
      withSQLConf(
          SQLConf.IGNORE_MISSING_FILES.key -> "false",
          SQLConf.IGNORE_CORRUPT_FILES.key -> "false") {
        withTempView("ws") {
          spark.read.option("ignoreCorruptFiles", "true").parquet(path.getCanonicalPath)
            .createOrReplaceTempView("ws")
          withViews("ids" -> values("id", "(1), (2), (3)")) {
            checkRewrite(directQuery, Seq(Row(1)), joinsWhenDisabled = 2, joinsWhenEnabled = 2)
          }
        }
      }
    }
  }

  test("an RDD-backed relation is not repeatable and is not rewritten") {
    // A relation whose rows come from arbitrary user code holds no expressions at all, so it is
    // trivially `deterministic` while nothing keeps two reads returning the same rows -- the case
    // where folding two evaluations into one is unsound. It declares no repeatability, so the join
    // count has to stay at two.
    val schema = StructType(Seq(StructField("k", IntegerType), StructField("v", IntegerType)))
    val rows = spark.sparkContext.parallelize(Seq(Row(1, 10), Row(1, 20), Row(2, 30), Row(2, 30)))
    withTempView("ws") {
      spark.createDataFrame(rows, schema).createOrReplaceTempView("ws")
      withViews("ids" -> values("id", "(1), (2), (3)")) {
        checkRewrite(directQuery, Seq(Row(1)), joinsWhenDisabled = 2, joinsWhenEnabled = 2)
      }
    }
  }
}
