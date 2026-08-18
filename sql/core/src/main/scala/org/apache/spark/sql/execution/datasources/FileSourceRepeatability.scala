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
package org.apache.spark.sql.execution.datasources

import org.apache.spark.sql.catalyst.FileSourceOptions
import org.apache.spark.sql.catalyst.util.CaseInsensitiveMap
import org.apache.spark.sql.execution.datasources.binaryfile.BinaryFileFormat
import org.apache.spark.sql.execution.datasources.csv.CSVFileFormat
import org.apache.spark.sql.execution.datasources.json.JsonFileFormat
import org.apache.spark.sql.execution.datasources.orc.OrcFileFormat
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.execution.datasources.text.TextFileFormat
import org.apache.spark.sql.sources.BaseRelation

/**
 * The file-source ingredients of relation repeatability: whether the format decodes the file's own
 * bytes and nothing else, and whether reads of the relation are strict rather than best-effort.
 *
 * File formats are trusted individually: a trusted reader decodes the file's own bytes, so
 * re-reading an unchanged file yields the same rows. Formats outside the list, and every non-file
 * relation, fail closed -- notably connector and RDD-backed relations, whose contents are produced
 * by code Spark cannot inspect.
 *
 * These are necessary conditions for a caller that has to decide whether re-reading a relation
 * returns the same rows, not a complete guarantee for a whole plan. Callers add whatever else their
 * own decision needs: [[LogicalRelation.isOutputRepeatable]] excludes streaming sources, and
 * [[org.apache.spark.sql.execution.columnar.InMemoryRelation]] additionally requires safe
 * expressions in the analyzed, optimized and physical plans, a supported operator set, a
 * non-indeterminate RDD lineage and strict file reads observed at execution time.
 */
private[sql] object FileSourceRepeatability {

  private val trustedFileFormatClasses: Set[Class[_ <: FileFormat]] = Set(
    classOf[BinaryFileFormat],
    classOf[CSVFileFormat],
    classOf[JsonFileFormat],
    classOf[OrcFileFormat],
    classOf[ParquetFileFormat],
    classOf[TextFileFormat])

  private val trustedExternalFileFormatNames = Set(
    "org.apache.spark.sql.avro.AvroFileFormat",
    "org.apache.spark.sql.hive.orc.OrcFileFormat")

  /** Whether reading a file of this format returns exactly the rows the file holds. */
  def isTrustedFileFormat(fileFormat: FileFormat): Boolean = {
    val fileFormatClass = fileFormat.getClass
    trustedFileFormatClasses.contains(fileFormatClass) ||
      trustedExternalFileFormatNames.contains(fileFormatClass.getName)
  }

  /**
   * Whether reads of a relation with these options are strict. A best-effort read may silently skip
   * a missing or corrupt file, so two reads of the same relation can return different rows.
   */
  def hasStrictReads(options: Map[String, String]): Boolean = {
    val effectiveOptions = new FileSourceOptions(CaseInsensitiveMap(options))
    !effectiveOptions.ignoreMissingFiles && !effectiveOptions.ignoreCorruptFiles
  }

  /**
   * Whether the relation itself keeps two reads of it returning the same rows: only a trusted file
   * format read strictly does.
   */
  def isRepeatable(relation: BaseRelation): Boolean = relation match {
    case fileRelation: HadoopFsRelation =>
      isTrustedFileFormat(fileRelation.fileFormat) && hasStrictReads(fileRelation.options)
    case _ => false
  }
}
