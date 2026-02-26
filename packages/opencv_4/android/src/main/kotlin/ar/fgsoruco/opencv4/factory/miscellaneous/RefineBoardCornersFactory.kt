package ar.fgsoruco.opencv4.factory.miscellaneous

import io.flutter.plugin.common.MethodChannel
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.FileInputStream
import java.io.InputStream
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class RefineBoardCornersFactory {
    companion object {
        fun process(
            pathType: Int,
            pathString: String,
            data: ByteArray,
            corners: ArrayList<Double>,
            roiPaddingRatio: Double,
            minAreaRatio: Double,
            approxEpsilonRatio: Double,
            cannyLow: Int,
            cannyHigh: Int,
            subPixWinSize: Int,
            maxCandidates: Int,
            result: MethodChannel.Result
        ) {
            try {
                result.success(
                    refine(
                        pathType = pathType,
                        pathString = pathString,
                        data = data,
                        corners = corners,
                        roiPaddingRatio = roiPaddingRatio,
                        minAreaRatio = minAreaRatio,
                        approxEpsilonRatio = approxEpsilonRatio,
                        cannyLow = cannyLow,
                        cannyHigh = cannyHigh,
                        subPixWinSize = subPixWinSize,
                        maxCandidates = maxCandidates
                    )
                )
            } catch (e: Exception) {
                val fallback = hashMapOf<String, Any>(
                    "ok" to false,
                    "reason" to "native_exception(${e.message ?: "unknown"})",
                    "corners" to corners
                )
                result.success(fallback)
            }
        }

        private fun refine(
            pathType: Int,
            pathString: String,
            data: ByteArray,
            corners: ArrayList<Double>,
            roiPaddingRatio: Double,
            minAreaRatio: Double,
            approxEpsilonRatio: Double,
            cannyLow: Int,
            cannyHigh: Int,
            subPixWinSize: Int,
            maxCandidates: Int
        ): Map<String, Any> {
            val fallback = hashMapOf<String, Any>(
                "ok" to false,
                "reason" to "fallback",
                "corners" to corners
            )
            if (corners.size < 8) {
                fallback["reason"] = "fallback_invalid_input_corners"
                return fallback
            }

            val src = decode(pathType, pathString, data)
            if (src.empty()) {
                fallback["reason"] = "fallback_decode_failed"
                return fallback
            }
            val width = src.cols()
            val height = src.rows()
            if (width <= 2 || height <= 2) {
                fallback["reason"] = "fallback_invalid_image_size"
                return fallback
            }

            val gray = Mat()
            when (src.channels()) {
                1 -> src.copyTo(gray)
                4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
                else -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            }

            val initialOrdered = orderPoints(
                arrayOf(
                    Point(corners[0], corners[1]),
                    Point(corners[2], corners[3]),
                    Point(corners[4], corners[5]),
                    Point(corners[6], corners[7]),
                )
            )
            if (!isReasonableQuad(initialOrdered, width, height)) {
                fallback["reason"] = "fallback_invalid_initial_quad"
                return fallback
            }

            val roiRect = buildRoiRect(
                points = initialOrdered,
                imageWidth = width,
                imageHeight = height,
                padRatio = roiPaddingRatio
            )
            if (roiRect.width < 20 || roiRect.height < 20) {
                fallback["reason"] = "fallback_roi_too_small(${roiRect.width}x${roiRect.height})"
                return fallback
            }

            val roiGray = Mat(gray, roiRect)
            val roiBlur = Mat()
            Imgproc.GaussianBlur(roiGray, roiBlur, Size(5.0, 5.0), 0.0)

            val binaryMaps = buildBinaryMaps(
                roiBlur = roiBlur,
                cannyLow = cannyLow,
                cannyHigh = cannyHigh
            )

            val roiArea = (roiRect.width * roiRect.height).toDouble()
            val minContourArea = max(30.0, roiArea * max(0.001, minAreaRatio))
            var best: Array<Point>? = null
            var bestScore = Double.NEGATIVE_INFINITY
            var bestLabel = "none"
            var candidatesTried = 0
            var accepted = 0

            outer@ for ((label, mask) in binaryMaps) {
                val contours = ArrayList<MatOfPoint>()
                Imgproc.findContours(
                    mask.clone(),
                    contours,
                    Mat(),
                    Imgproc.RETR_LIST,
                    Imgproc.CHAIN_APPROX_SIMPLE
                )
                val sorted = contours.sortedByDescending { contour ->
                    Imgproc.contourArea(contour)
                }
                for (contour in sorted) {
                    if (candidatesTried >= max(10, maxCandidates)) {
                        break@outer
                    }
                    val area = Imgproc.contourArea(contour)
                    if (area < minContourArea) {
                        continue
                    }
                    val contour2f = MatOfPoint2f(*contour.toArray())
                    val perimeter = Imgproc.arcLength(contour2f, true)
                    if (perimeter < 25.0) {
                        continue
                    }
                    val localCandidates = buildLocalQuadCandidates(
                        contour2f = contour2f,
                        perimeter = perimeter,
                        approxEpsilonRatio = approxEpsilonRatio
                    )

                    for ((candidateType, localQuad) in localCandidates) {
                        if (candidatesTried >= max(10, maxCandidates)) {
                            break@outer
                        }
                        candidatesTried += 1
                        val global = Array(4) { i ->
                            Point(localQuad[i].x + roiRect.x, localQuad[i].y + roiRect.y)
                        }
                        val ordered = orderPoints(global)
                        if (!isReasonableQuad(ordered, width, height)) {
                            continue
                        }
                        accepted += 1
                        val score = scoreCandidate(
                            candidate = ordered,
                            initial = initialOrdered,
                            imageWidth = width,
                            imageHeight = height,
                            gray = gray
                        )
                        if (score > bestScore) {
                            bestScore = score
                            best = ordered
                            bestLabel = "$label/$candidateType"
                        }
                    }
                }
            }

            val warpCandidates = buildWarpInnerQuadCandidates(
                gray = gray,
                initialOrdered = initialOrdered
            )
            for ((label, candidate) in warpCandidates) {
                if (candidatesTried >= max(10, maxCandidates)) {
                    break
                }
                candidatesTried += 1
                val ordered = orderPoints(candidate)
                if (!isReasonableQuad(ordered, width, height)) {
                    continue
                }
                accepted += 1
                val score = scoreCandidate(
                    candidate = ordered,
                    initial = initialOrdered,
                    imageWidth = width,
                    imageHeight = height,
                    gray = gray
                )
                if (score > bestScore) {
                    bestScore = score
                    best = ordered
                    bestLabel = "warp/$label"
                }
            }

            if (best == null) {
                fallback["reason"] = "fallback_no_quad(tried=$candidatesTried accepted=$accepted)"
                return fallback
            }

            val refined = refineSubPixel(
                roiGrayBlurred = roiBlur,
                roiRect = roiRect,
                orderedPoints = best,
                subPixWinSize = subPixWinSize
            )
            val refinedOrdered = orderPoints(refined)
            val refinedScore = scoreCandidate(
                candidate = refinedOrdered,
                initial = initialOrdered,
                imageWidth = width,
                imageHeight = height,
                gray = gray
            )

            val finalPoints = if (
                isReasonableQuad(refinedOrdered, width, height) &&
                refinedScore >= bestScore - 0.02
            ) {
                refinedOrdered
            } else {
                best
            }
            val finalScore = if (finalPoints === refinedOrdered) refinedScore else bestScore

            return hashMapOf<String, Any>(
                "ok" to true,
                "reason" to "native_ok(map=$bestLabel score=${"%.4f".format(finalScore)})",
                "corners" to flatten(finalPoints),
                "score" to finalScore
            )
        }

        private fun decode(pathType: Int, pathString: String, data: ByteArray): Mat {
            return try {
                when (pathType) {
                    1 -> {
                        val filename = pathString.replace("file://", "")
                        val inputStream: InputStream = FileInputStream(filename)
                        inputStream.use { _ -> Imgcodecs.imread(filename, Imgcodecs.IMREAD_UNCHANGED) }
                    }
                    else -> Imgcodecs.imdecode(MatOfByte(*data), Imgcodecs.IMREAD_UNCHANGED)
                }
            } catch (_: Exception) {
                Mat()
            }
        }

        private fun buildBinaryMaps(
            roiBlur: Mat,
            cannyLow: Int,
            cannyHigh: Int
        ): List<Pair<String, Mat>> {
            val maps = ArrayList<Pair<String, Mat>>()

            val canny = Mat()
            Imgproc.Canny(
                roiBlur,
                canny,
                max(5, cannyLow).toDouble(),
                max(cannyLow + 5, cannyHigh).toDouble()
            )
            maps.add(Pair("canny", canny))
            val cannyDilated = Mat()
            Imgproc.dilate(canny, cannyDilated, Mat(), Point(-1.0, -1.0), 1)
            maps.add(Pair("canny_dilate", cannyDilated))

            val otsu = Mat()
            Imgproc.threshold(
                roiBlur,
                otsu,
                0.0,
                255.0,
                Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
            )
            maps.add(Pair("otsu", otsu))

            val adaptive = Mat()
            Imgproc.adaptiveThreshold(
                roiBlur,
                adaptive,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY,
                17,
                4.0
            )
            maps.add(Pair("adaptive", adaptive))
            val adaptiveClosed = Mat()
            Imgproc.morphologyEx(
                adaptive,
                adaptiveClosed,
                Imgproc.MORPH_CLOSE,
                Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            )
            maps.add(Pair("adaptive_close", adaptiveClosed))

            val cannyInv = Mat()
            Core.bitwise_not(canny, cannyInv)
            maps.add(Pair("canny_inv", cannyInv))

            val otsuInv = Mat()
            Core.bitwise_not(otsu, otsuInv)
            maps.add(Pair("otsu_inv", otsuInv))

            val adaptiveInv = Mat()
            Core.bitwise_not(adaptive, adaptiveInv)
            maps.add(Pair("adaptive_inv", adaptiveInv))
            return maps
        }

        private fun buildWarpInnerQuadCandidates(
            gray: Mat,
            initialOrdered: Array<Point>
        ): List<Pair<String, Array<Point>>> {
            val out = ArrayList<Pair<String, Array<Point>>>()
            if (gray.empty()) {
                return out
            }
            try {
                val size = 256.0
                val n = size.toInt()
                val srcPts = MatOfPoint2f(*initialOrdered)
                val dstPts = MatOfPoint2f(
                    Point(0.0, 0.0),
                    Point(size - 1.0, 0.0),
                    Point(size - 1.0, size - 1.0),
                    Point(0.0, size - 1.0)
                )
                val toWarp = Imgproc.getPerspectiveTransform(srcPts, dstPts)
                val toSrc = Imgproc.getPerspectiveTransform(dstPts, srcPts)

                val warped = Mat()
                Imgproc.warpPerspective(gray, warped, toWarp, Size(size, size))
                val warpedBlur = Mat()
                Imgproc.GaussianBlur(warped, warpedBlur, Size(5.0, 5.0), 0.0)
                val edges = Mat()
                Imgproc.Canny(warpedBlur, edges, 45.0, 150.0)

                val colScore = DoubleArray(n)
                val rowScore = DoubleArray(n)
                for (y in 0 until n) {
                    for (x in 0 until n) {
                        val v = edges.get(y, x)[0]
                        colScore[x] += v
                        rowScore[y] += v
                    }
                }
                val colSmooth = smoothScores(colScore, radius = 4)
                val rowSmooth = smoothScores(rowScore, radius = 4)

                val left = findPeakIndex(colSmooth, (n * 0.08).toInt(), (n * 0.62).toInt())
                val right = findPeakIndex(colSmooth, (n * 0.38).toInt(), (n * 0.95).toInt())
                val top = findPeakIndex(rowSmooth, (n * 0.08).toInt(), (n * 0.62).toInt())
                val bottom = findPeakIndex(rowSmooth, (n * 0.38).toInt(), (n * 0.95).toInt())

                if (left >= 0 && right >= 0 && top >= 0 && bottom >= 0) {
                    if (right - left >= (n * 0.32) && bottom - top >= (n * 0.32)) {
                        val base = reprojectCanonicalQuad(
                            toSrc = toSrc,
                            left = left.toDouble(),
                            top = top.toDouble(),
                            right = right.toDouble(),
                            bottom = bottom.toDouble()
                        )
                        if (base != null) {
                            out.add(Pair("profile", base))
                        }

                        val pad = max(2, (n * 0.015).toInt()).toDouble()
                        val expanded = reprojectCanonicalQuad(
                            toSrc = toSrc,
                            left = max(0.0, left - pad),
                            top = max(0.0, top - pad),
                            right = min(size - 1.0, right + pad),
                            bottom = min(size - 1.0, bottom + pad)
                        )
                        if (expanded != null) {
                            out.add(Pair("profile_expanded", expanded))
                        }
                    }
                }

                edges.release()
                warpedBlur.release()
                warped.release()
                toWarp.release()
                toSrc.release()
            } catch (_: Exception) {
                return out
            }
            return out
        }

        private fun smoothScores(input: DoubleArray, radius: Int): DoubleArray {
            if (input.isEmpty() || radius <= 0) {
                return input
            }
            val out = DoubleArray(input.size)
            for (i in input.indices) {
                var sum = 0.0
                var count = 0
                val start = max(0, i - radius)
                val end = min(input.size - 1, i + radius)
                for (j in start..end) {
                    sum += input[j]
                    count += 1
                }
                out[i] = if (count > 0) sum / count else input[i]
            }
            return out
        }

        private fun findPeakIndex(scores: DoubleArray, start: Int, end: Int): Int {
            if (scores.isEmpty()) {
                return -1
            }
            val s = start.coerceIn(0, scores.size - 1)
            val e = end.coerceIn(s, scores.size - 1)
            var bestIdx = -1
            var bestValue = Double.NEGATIVE_INFINITY
            for (i in s..e) {
                if (scores[i] > bestValue) {
                    bestValue = scores[i]
                    bestIdx = i
                }
            }
            return bestIdx
        }

        private fun reprojectCanonicalQuad(
            toSrc: Mat,
            left: Double,
            top: Double,
            right: Double,
            bottom: Double
        ): Array<Point>? {
            return try {
                val src = MatOfPoint2f(
                    Point(left, top),
                    Point(right, top),
                    Point(right, bottom),
                    Point(left, bottom)
                )
                val out = MatOfPoint2f()
                Core.perspectiveTransform(src, out, toSrc)
                val points = out.toArray()
                if (points.size == 4) points else null
            } catch (_: Exception) {
                null
            }
        }

        private fun buildLocalQuadCandidates(
            contour2f: MatOfPoint2f,
            perimeter: Double,
            approxEpsilonRatio: Double
        ): List<Pair<String, Array<Point>>> {
            val candidates = ArrayList<Pair<String, Array<Point>>>()
            val epsilonBase = perimeter * max(0.005, approxEpsilonRatio)
            val epsilonScales = doubleArrayOf(0.8, 1.0, 1.25)
            for (scale in epsilonScales) {
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(
                    contour2f,
                    approx,
                    epsilonBase * scale,
                    true
                )
                val approxPoints = approx.toArray()
                if (approxPoints.size != 4) {
                    continue
                }
                val contourInt = MatOfPoint(*approxPoints)
                if (!Imgproc.isContourConvex(contourInt)) {
                    continue
                }
                candidates.add(Pair("approx_${scale}", approxPoints))
            }

            val rot: RotatedRect = Imgproc.minAreaRect(contour2f)
            val rotPoints = Array(4) { Point() }
            rot.points(rotPoints)
            candidates.add(Pair("minrect", rotPoints))
            return candidates
        }

        private fun refineSubPixel(
            roiGrayBlurred: Mat,
            roiRect: Rect,
            orderedPoints: Array<Point>,
            subPixWinSize: Int
        ): Array<Point> {
            return try {
                val local = orderedPoints.map { p ->
                    Point(
                        (p.x - roiRect.x).coerceIn(0.0, max(0.0, roiGrayBlurred.cols() - 1.0)),
                        (p.y - roiRect.y).coerceIn(0.0, max(0.0, roiGrayBlurred.rows() - 1.0))
                    )
                }
                val mat = MatOfPoint2f(*local.toTypedArray())
                val win = max(2, min(12, subPixWinSize))
                Imgproc.cornerSubPix(
                    roiGrayBlurred,
                    mat,
                    Size(win.toDouble(), win.toDouble()),
                    Size(-1.0, -1.0),
                    TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.01)
                )
                val refinedLocal = mat.toArray()
                Array(4) { i ->
                    val p = refinedLocal[i]
                    Point(p.x + roiRect.x, p.y + roiRect.y)
                }
            } catch (_: Exception) {
                orderedPoints
            }
        }

        private fun flatten(points: Array<Point>): ArrayList<Double> {
            val out = ArrayList<Double>(8)
            for (p in points) {
                out.add(p.x)
                out.add(p.y)
            }
            return out
        }

        private fun buildRoiRect(
            points: Array<Point>,
            imageWidth: Int,
            imageHeight: Int,
            padRatio: Double
        ): Rect {
            val minX = points.minOf { it.x }
            val maxX = points.maxOf { it.x }
            val minY = points.minOf { it.y }
            val maxY = points.maxOf { it.y }
            val w = max(1.0, maxX - minX)
            val h = max(1.0, maxY - minY)
            val pad = max(8.0, max(w, h) * max(0.0, padRatio))
            val left = floor(minX - pad).toInt().coerceIn(0, imageWidth - 1)
            val top = floor(minY - pad).toInt().coerceIn(0, imageHeight - 1)
            val right = ceil(maxX + pad).toInt().coerceIn(0, imageWidth - 1)
            val bottom = ceil(maxY + pad).toInt().coerceIn(0, imageHeight - 1)
            val width = max(1, right - left + 1)
            val height = max(1, bottom - top + 1)
            return Rect(left, top, width, height)
        }

        private fun orderPoints(points: Array<Point>): Array<Point> {
            var tl = points[0]
            var tr = points[0]
            var br = points[0]
            var bl = points[0]
            var tlScore = Double.POSITIVE_INFINITY
            var brScore = Double.NEGATIVE_INFINITY
            var trScore = Double.NEGATIVE_INFINITY
            var blScore = Double.POSITIVE_INFINITY

            for (p in points) {
                val sum = p.x + p.y
                val diff = p.x - p.y
                if (sum < tlScore) {
                    tlScore = sum
                    tl = p
                }
                if (sum > brScore) {
                    brScore = sum
                    br = p
                }
                if (diff > trScore) {
                    trScore = diff
                    tr = p
                }
                if (diff < blScore) {
                    blScore = diff
                    bl = p
                }
            }
            return arrayOf(tl, tr, br, bl)
        }

        private fun isReasonableQuad(points: Array<Point>, imageWidth: Int, imageHeight: Int): Boolean {
            if (points.size != 4) {
                return false
            }
            for (p in points) {
                if (p.x.isNaN() || p.y.isNaN()) {
                    return false
                }
                if (p.x < 0 || p.y < 0 || p.x > imageWidth - 1 || p.y > imageHeight - 1) {
                    return false
                }
            }

            val area = quadArea(points)
            val imageArea = max(1.0, (imageWidth * imageHeight).toDouble())
            val ratio = area / imageArea
            if (ratio < 0.02 || ratio > 0.995) {
                return false
            }

            val dTop = distance(points[0], points[1])
            val dRight = distance(points[1], points[2])
            val dBottom = distance(points[2], points[3])
            val dLeft = distance(points[3], points[0])
            val minSide = min(min(dTop, dRight), min(dBottom, dLeft))
            if (minSide < 14.0) {
                return false
            }
            val avgH = (dTop + dBottom) * 0.5
            val avgV = (dLeft + dRight) * 0.5
            val aspect = max(avgH, avgV) / max(1e-6, min(avgH, avgV))
            if (aspect > 2.1) {
                return false
            }
            return true
        }

        private fun scoreCandidate(
            candidate: Array<Point>,
            initial: Array<Point>,
            imageWidth: Int,
            imageHeight: Int,
            gray: Mat
        ): Double {
            val candArea = max(1e-6, quadArea(candidate))
            val initArea = max(1e-6, quadArea(initial))
            val areaRatio = candArea / initArea
            val areaScore = (1.0 - (abs(ln(areaRatio)) / 1.4)).coerceIn(0.0, 1.0)

            val avgInitSide = averageSide(initial)
            var d = 0.0
            for (i in 0 until 4) {
                d += distance(candidate[i], initial[i])
            }
            d /= 4.0
            val proximity = (1.0 - (d / max(1e-6, avgInitSide * 0.80))).coerceIn(0.0, 1.0)
            val checker = checkerboardScoreAfterWarp(gray, candidate)
            val edgeSupport = edgeSupportMetrics(gray, candidate)
            val edgeMean = edgeSupport.first
            val edgeBalance = edgeSupport.second
            val shapeScore = quadShapeScore(candidate)

            val imageArea = max(1.0, (imageWidth * imageHeight).toDouble())
            val areaPenalty = when {
                candArea / imageArea < 0.03 -> 0.35
                candArea / imageArea < 0.08 -> 0.70
                else -> 1.0
            }

            val margin = max(2.0, min(imageWidth, imageHeight) * 0.015)
            val nearMargin = margin * 2.0
            var borderPressure = 0.0
            for (p in candidate) {
                val dx = min(p.x, (imageWidth - 1) - p.x)
                val dy = min(p.y, (imageHeight - 1) - p.y)
                val edgeDistance = min(dx, dy)
                if (edgeDistance <= margin * 0.35) {
                    borderPressure += 1.0
                } else if (edgeDistance <= margin) {
                    borderPressure += 0.75
                } else if (edgeDistance <= nearMargin) {
                    borderPressure += 0.35
                }
            }
            val borderPenalty = (1.0 - ((borderPressure / 4.0) * 0.55)).coerceIn(0.50, 1.0)

            return ((checker * 0.34) +
                (edgeMean * 0.18) +
                (edgeBalance * 0.18) +
                (shapeScore * 0.15) +
                (proximity * 0.08) +
                (areaScore * 0.10)) *
                areaPenalty *
                borderPenalty
        }

        private fun edgeSupportMetrics(gray: Mat, candidate: Array<Point>): Pair<Double, Double> {
            if (gray.empty() || candidate.size != 4 || gray.cols() < 3 || gray.rows() < 3) {
                return Pair(0.0, 0.0)
            }
            val width = gray.cols()
            val height = gray.rows()
            val edgeScores = DoubleArray(4)
            for (i in 0 until 4) {
                val a = candidate[i]
                val b = candidate[(i + 1) % 4]
                val edgeLen = max(1.0, distance(a, b))
                val samples = max(6, min(24, (edgeLen / 16.0).toInt()))
                val nx = -((b.y - a.y) / edgeLen)
                val ny = (b.x - a.x) / edgeLen
                var edgeSum = 0.0
                var edgeCount = 0
                for (s in 0..samples) {
                    val t = s.toDouble() / samples.toDouble()
                    val x = a.x + (b.x - a.x) * t
                    val y = a.y + (b.y - a.y) * t
                    val px = x.roundToInt().coerceIn(1, width - 2)
                    val py = y.roundToInt().coerceIn(1, height - 2)
                    val gx = gray.get(py, px + 1)[0] - gray.get(py, px - 1)[0]
                    val gy = gray.get(py + 1, px)[0] - gray.get(py - 1, px)[0]
                    val grad = sqrt((gx * gx) + (gy * gy))
                    val normal = abs(gx * nx + gy * ny)
                    edgeSum += (normal * 0.65) + (grad * 0.35)
                    edgeCount += 1
                }
                if (edgeCount > 0) {
                    edgeScores[i] = edgeSum / edgeCount.toDouble()
                }
            }
            val meanRaw = edgeScores.average()
            if (meanRaw <= 1e-6) {
                return Pair(0.0, 0.0)
            }
            val minRaw = edgeScores.minOrNull() ?: 0.0
            val meanScore = (meanRaw / 96.0).coerceIn(0.0, 1.0)
            val balanceScore = (minRaw / meanRaw).coerceIn(0.0, 1.0)
            return Pair(meanScore, balanceScore)
        }

        private fun quadShapeScore(candidate: Array<Point>): Double {
            if (candidate.size != 4) {
                return 0.0
            }
            val top = distance(candidate[0], candidate[1])
            val right = distance(candidate[1], candidate[2])
            val bottom = distance(candidate[2], candidate[3])
            val left = distance(candidate[3], candidate[0])
            val minSide = min(min(top, right), min(bottom, left))
            if (minSide < 8.0) {
                return 0.0
            }

            val oppH = min(top, bottom) / max(top, bottom)
            val oppV = min(left, right) / max(left, right)
            val oppositeScore = ((oppH + oppV) * 0.5).coerceIn(0.0, 1.0)

            val avgH = (top + bottom) * 0.5
            val avgV = (left + right) * 0.5
            val aspect = max(avgH, avgV) / max(1e-6, min(avgH, avgV))
            val aspectScore = (1.0 - ((aspect - 1.0) / 1.2)).coerceIn(0.0, 1.0)

            val a0 = cornerAngle(candidate[3], candidate[0], candidate[1])
            val a1 = cornerAngle(candidate[0], candidate[1], candidate[2])
            val a2 = cornerAngle(candidate[1], candidate[2], candidate[3])
            val a3 = cornerAngle(candidate[2], candidate[3], candidate[0])
            val avgDeviation = (
                abs(a0 - 90.0) +
                    abs(a1 - 90.0) +
                    abs(a2 - 90.0) +
                    abs(a3 - 90.0)
                ) / 4.0
            val angleScore = (1.0 - (avgDeviation / 55.0)).coerceIn(0.0, 1.0)

            return (
                (oppositeScore * 0.45) +
                    (aspectScore * 0.25) +
                    (angleScore * 0.30)
                ).coerceIn(0.0, 1.0)
        }

        private fun cornerAngle(a: Point, b: Point, c: Point): Double {
            val v1x = a.x - b.x
            val v1y = a.y - b.y
            val v2x = c.x - b.x
            val v2y = c.y - b.y
            val n1 = sqrt((v1x * v1x) + (v1y * v1y))
            val n2 = sqrt((v2x * v2x) + (v2y * v2y))
            if (n1 < 1e-6 || n2 < 1e-6) {
                return 90.0
            }
            val dot = (v1x * v2x) + (v1y * v2y)
            val cos = (dot / (n1 * n2)).coerceIn(-1.0, 1.0)
            return acos(cos) * 180.0 / Math.PI
        }

        private fun checkerboardScoreAfterWarp(gray: Mat, candidate: Array<Point>): Double {
            return try {
                val size = 192.0
                val srcPts = MatOfPoint2f(*candidate)
                val dstPts = MatOfPoint2f(
                    Point(0.0, 0.0),
                    Point(size - 1.0, 0.0),
                    Point(size - 1.0, size - 1.0),
                    Point(0.0, size - 1.0)
                )
                val transform = Imgproc.getPerspectiveTransform(srcPts, dstPts)
                val warped = Mat()
                Imgproc.warpPerspective(gray, warped, transform, Size(size, size))

                val cell = size / 8.0
                val means = DoubleArray(64)
                var evenSum = 0.0
                var oddSum = 0.0
                var evenCount = 0
                var oddCount = 0
                for (row in 0 until 8) {
                    for (col in 0 until 8) {
                        val x0 = floor(col * cell + cell * 0.2).toInt().coerceIn(0, warped.cols() - 1)
                        val y0 = floor(row * cell + cell * 0.2).toInt().coerceIn(0, warped.rows() - 1)
                        val x1 = floor((col + 1) * cell - cell * 0.2).toInt().coerceIn(x0 + 1, warped.cols())
                        val y1 = floor((row + 1) * cell - cell * 0.2).toInt().coerceIn(y0 + 1, warped.rows())
                        val sub = warped.submat(y0, y1, x0, x1)
                        val m = Core.mean(sub).`val`[0]
                        sub.release()
                        val idx = row * 8 + col
                        means[idx] = m
                        if (((row + col) and 1) == 0) {
                            evenSum += m
                            evenCount += 1
                        } else {
                            oddSum += m
                            oddCount += 1
                        }
                    }
                }
                if (evenCount == 0 || oddCount == 0) {
                    warped.release()
                    transform.release()
                    return 0.0
                }
                val evenMean = evenSum / evenCount
                val oddMean = oddSum / oddCount
                val parity = (abs(evenMean - oddMean) / 96.0).coerceIn(0.0, 1.0)

                var adjacency = 0.0
                var adjacencyCount = 0
                for (row in 0 until 8) {
                    for (col in 0 until 7) {
                        val a = means[row * 8 + col]
                        val b = means[row * 8 + col + 1]
                        adjacency += abs(a - b)
                        adjacencyCount += 1
                    }
                }
                for (col in 0 until 8) {
                    for (row in 0 until 7) {
                        val a = means[row * 8 + col]
                        val b = means[(row + 1) * 8 + col]
                        adjacency += abs(a - b)
                        adjacencyCount += 1
                    }
                }
                val adjacencyScore = if (adjacencyCount == 0) {
                    0.0
                } else {
                    ((adjacency / adjacencyCount) / 96.0).coerceIn(0.0, 1.0)
                }
                warped.release()
                transform.release()
                ((parity * 0.70) + (adjacencyScore * 0.30)).coerceIn(0.0, 1.0)
            } catch (_: Exception) {
                0.0
            }
        }

        private fun averageSide(points: Array<Point>): Double {
            if (points.size != 4) {
                return 1.0
            }
            return (
                distance(points[0], points[1]) +
                distance(points[1], points[2]) +
                distance(points[2], points[3]) +
                distance(points[3], points[0])
            ) / 4.0
        }

        private fun quadArea(points: Array<Point>): Double {
            var sum = 0.0
            for (i in 0 until 4) {
                val p = points[i]
                val n = points[(i + 1) % 4]
                sum += p.x * n.y - n.x * p.y
            }
            return abs(sum) * 0.5
        }

        private fun distance(a: Point, b: Point): Double {
            return sqrt((a.x - b.x).pow(2.0) + (a.y - b.y).pow(2.0))
        }
    }
}
