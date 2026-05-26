import numpy as np

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape
import labelme.utils


# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor
CURSOR_BRUSH = QtCore.Qt.BlankCursor   # 브러시 모드: 시스템 커서 숨기고 직접 그림


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(bool)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    edgeSelected = QtCore.Signal(bool)
    maskToPolygonRequest = QtCore.Signal()
    brushSizeChanged = QtCore.Signal(int)   # 브러시 크기 변경 시 슬라이더 동기화용

    CREATE, EDIT, BRUSH = 0, 1, 2

    # polygon, rectangle, line, or point
    _createMode = 'polygon'

    _fill_drawing = False

    # 브러시 모드: 'draw' or 'erase'
    _brushMode = 'draw'
    _brushSize = 20

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 11.0)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShape = None  # save the selected shape here
        self.selectedShapeCopy = None
        self.lineColor = QtGui.QColor(0, 0, 255)
        self.line = Shape(line_color=self.lineColor)
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.movingShape = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT

        # 브러시 마스크 레이어
        self.maskImage = None          # QImage (ARGB32)
        self.maskColor = QtGui.QColor(255, 0, 0, 120)  # 반투명 빨강
        self._brushPainting = False    # 마우스 드래그 중 여부

        # Menus:
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

    # ------------------------------------------------------------------ #
    #  브러시 관련 프로퍼티
    # ------------------------------------------------------------------ #

    @property
    def brushMode(self):
        return self._brushMode

    @brushMode.setter
    def brushMode(self, value):
        assert value in ('draw', 'erase')
        self._brushMode = value

    @property
    def brushSize(self):
        return self._brushSize

    @brushSize.setter
    def brushSize(self, value):
        self._brushSize = max(1, int(value))

    def isBrushing(self):
        return self.mode == self.BRUSH

    def setBrushMode(self, brushType='draw'):
        """BRUSH 모드로 전환 + draw/erase 설정."""
        self.mode = self.BRUSH
        self._brushMode = brushType
        self.unHighlight()
        self.deSelectShape()
        self._ensureMask()

    def _ensureMask(self):
        """pixmap 크기에 맞는 마스크 이미지가 없으면 생성."""
        if self.pixmap and not self.pixmap.isNull():
            size = self.pixmap.size()
            if (self.maskImage is None or
                    self.maskImage.size() != size):
                self.maskImage = QtGui.QImage(
                    size, QtGui.QImage.Format_ARGB32)
                self.maskImage.fill(QtCore.Qt.transparent)

    def clearMask(self):
        """마스크 전체 초기화."""
        if self.maskImage is not None:
            self.maskImage.fill(QtCore.Qt.transparent)
            self.update()

    def hasMask(self):
        """마스크에 칠해진 픽셀이 있는지 여부."""
        if self.maskImage is None:
            return False
        try:
            arr = self._maskToNumpy()
            return bool(arr[:, :, 3].any())
        except Exception:
            return False

    def _maskToNumpy(self):
        """QImage → numpy ARGB 배열로 변환."""
        img = self.maskImage.convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(h * w * 4)
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

    def maskToPolygonPoints(self, simplify_epsilon=2.0):
        """
        현재 마스크를 OpenCV contour로 변환하여
        가장 큰 윤곽선의 포인트 리스트를 반환.
        반환값: [(x, y), ...] 또는 None
        """
        if self.maskImage is None:
            return None
        try:
            import cv2
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, 'Error', 'opencv-python 패키지가 필요합니다.\n'
                               'pip install opencv-python')
            return None

        # QImage → numpy (알파 채널 기준 마스크)
        arr = self._maskToNumpy()
        alpha = arr[:, :, 3]
        binary = (alpha > 0).astype(np.uint8) * 255

        # 윤곽선 추출
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 가장 큰 윤곽선 선택
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            return None

        # 폴리곤 단순화 (점 수 줄이기)
        approx = cv2.approxPolyDP(contour, simplify_epsilon, True)
        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        return points if len(points) >= 3 else None

    # ------------------------------------------------------------------ #
    #  기존 메서드
    # ------------------------------------------------------------------ #

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'circle',
           'line', 'point', 'linestrip']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.storeShapes()
        self.repaint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hVertex = self.hShape = None

    def selectedVertex(self):
        return self.hVertex is not None

    # ------------------------------------------------------------------ #
    #  브러시 페인팅 내부 메서드
    # ------------------------------------------------------------------ #

    def _paintBrushLine(self, p1, p2):
        """두 점 사이를 보간하여 부드럽게 칠함 (QPainter 1회만 생성)."""
        import math
        self._ensureMask()
        if self.maskImage is None:
            return

        dx = float(p2.x()) - float(p1.x())
        dy = float(p2.y()) - float(p1.y())
        dist = math.sqrt(dx * dx + dy * dy)
        step_size = max(1, self._brushSize // 4)
        steps = max(1, int(dist / step_size))

        painter = QtGui.QPainter(self.maskImage)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        r = max(1, self._brushSize // 2)

        if self._brushMode == 'draw':
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode_Source)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(self.maskColor)
        else:
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode_Clear)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtCore.Qt.transparent)

        for i in range(steps + 1):
            t = i / steps
            x = float(p1.x()) + dx * t
            y = float(p1.y()) + dy * t
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)

        painter.end()
        self.update()

    # ------------------------------------------------------------------ #
    #  마우스 이벤트
    # ------------------------------------------------------------------ #

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())

        self.prevMovePoint = pos
        self.restoreCursor()

        # ── 브러시 모드 ──────────────────────────────────────────────────
        if self.isBrushing():
            self.overrideCursor(CURSOR_BRUSH)
            if self._brushPainting and (
                    QtCore.Qt.LeftButton & ev.buttons()):
                if not self.outOfPixmap(pos) and self.prevPoint is not None:
                    self._paintBrushLine(self.prevPoint, pos)
                self.prevPoint = pos
            else:
                self.update()  # 커서 미리보기 갱신
            return

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            color = self.lineColor
            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif len(self.current) > 1 and self.createMode == 'polygon' and\
                    self.closeEnough(pos, self.current[0]):
                pos = self.current[0]
                color = self.current.line_color
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ['polygon', 'linestrip']:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == 'rectangle':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'circle':
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == 'line':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'point':
                self.line.points = [self.current[0]]
                self.line.close()
            self.line.line_color = color
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapeCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShapeCopy, pos)
                self.repaint()
            elif self.selectedShape:
                self.selectedShapeCopy = self.selectedShape.copy()
                self.repaint()
            return

        # Polygon/Vertex moving.
        self.movingShape = False
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShape and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShape, pos)
                self.repaint()
                self.movingShape = True
            return

        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = index
                self.hShape = shape
                self.hEdge = index_edge
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip("Click & drag to move point")
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = None
                self.hShape = shape
                self.hEdge = index_edge
                self.setToolTip(
                    "Click & drag to move shape '%s'" % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape, self.hEdge = None, None, None
        self.edgeSelected.emit(self.hEdge is not None)

    def addPointToEdge(self):
        if (self.hShape is None and
                self.hEdge is None and
                self.prevMovePoint is None):
            return
        shape = self.hShape
        index = self.hEdge
        point = self.prevMovePoint
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())

        # ── 브러시 모드 ──────────────────────────────────────────────────
        if self.isBrushing():
            if ev.button() == QtCore.Qt.LeftButton:
                if not self.outOfPixmap(pos):
                    self._brushPainting = True
                    self.prevPoint = pos
                    self._paintBrushLine(pos, pos)  # 첫 점 찍기
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    if self.createMode == 'polygon':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ['rectangle', 'circle', 'line']:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == 'linestrip':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        if self.createMode == 'circle':
                            self.current.shape_type = 'circle'
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            else:
                self.selectShapePoint(pos)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            self.selectShapePoint(pos)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        # ── 브러시 모드 ──────────────────────────────────────────────────
        if self.isBrushing():
            if ev.button() == QtCore.Qt.LeftButton:
                self._brushPainting = False
            return

        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[bool(self.selectedShapeCopy)]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapeCopy:
                self.selectedShapeCopy = None
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton and self.selectedShape:
            self.overrideCursor(CURSOR_GRAB)
        if self.movingShape:
            self.storeShapes()
            self.shapeMoved.emit()

    def endMove(self, copy=False):
        assert self.selectedShape and self.selectedShapeCopy
        shape = self.selectedShapeCopy
        if copy:
            self.shapes.append(shape)
            self.selectedShape.selected = False
            self.selectedShape = shape
            self.repaint()
        else:
            shape.label = self.selectedShape.label
            self.deleteSelected()
            self.shapes.append(shape)
        self.storeShapes()
        self.selectedShapeCopy = None

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShape:
            self.setHiding(True)
            self.repaint()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        if self.canCloseShape() and len(self.current) > 3:
            self.current.popPoint()
            self.finalise()

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selectedShape = shape
        self.setHiding()
        self.selectionChanged.emit(True)
        self.update()

    def selectShapePoint(self, point):
        """Select the first shape created which contains this point."""
        self.deSelectShape()
        if self.selectedVertex():
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
            return
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                shape.selected = True
                self.selectedShape = shape
                self.calculateOffsets(shape, point)
                self.setHiding()
                self.selectionChanged.emit(True)
                return

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width() - 1) - point.x()
        y2 = (rect.y() + rect.height() - 1) - point.y()
        self.offsets = QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShape(self, shape, pos):
        if self.outOfPixmap(pos):
            return False
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(min(0, self.pixmap.width() - o2.x()),
                                 min(0, self.pixmap.height() - o2.y()))
        dp = pos - self.prevPoint
        if dp:
            shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShape:
            self.selectedShape.selected = False
            self.selectedShape = None
            self.setHiding(False)
            self.selectionChanged.emit(False)
            self.update()

    def deleteSelected(self):
        if self.selectedShape:
            shape = self.selectedShape
            self.shapes.remove(self.selectedShape)
            self.storeShapes()
            self.selectedShape = None
            self.update()
            return shape

    def copySelectedShape(self):
        if self.selectedShape:
            shape = self.selectedShape.copy()
            self.deSelectShape()
            self.shapes.append(shape)
            self.storeShapes()
            shape.selected = True
            self.selectedShape = shape
            self.boundedShiftShape(shape)
            return shape

    def boundedShiftShape(self, shape):
        point = shape[0]
        offset = QtCore.QPoint(2.0, 2.0)
        self.calculateOffsets(shape, point)
        self.prevPoint = point
        if not self.boundedMoveShape(shape, point - offset):
            self.boundedMoveShape(shape, point + offset)

    # ------------------------------------------------------------------ #
    #  렌더링
    # ------------------------------------------------------------------ #

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        # 브러시 마스크 오버레이 렌더링
        if self.maskImage is not None:
            p.drawImage(0, 0, self.maskImage)

        # 브러시 커서 미리보기
        if self.isBrushing():
            pos = self.prevMovePoint
            r = max(1, self._brushSize // 2)
            cx, cy = int(pos.x()), int(pos.y())

            if self._brushMode == 'draw':
                outer_color = QtGui.QColor(255, 80, 80, 230)
                inner_color = QtGui.QColor(255, 80, 80, 40)
            else:
                outer_color = QtGui.QColor(60, 140, 255, 230)
                inner_color = QtGui.QColor(60, 140, 255, 30)

            # 반투명 내부 채움
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(inner_color)
            p.drawEllipse(QtCore.QPoint(cx, cy), r, r)

            # 외곽선 (흰색 테두리 + 색상 테두리로 대비)
            pen_w = max(1, int(1.5 / self.scale))
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), pen_w + 1))
            p.drawEllipse(QtCore.QPoint(cx, cy), r, r)
            p.setPen(QtGui.QPen(outer_color, pen_w))
            p.drawEllipse(QtCore.QPoint(cx, cy), r, r)

            # 중심 십자 (작은 점)
            cs = max(2, int(3 / self.scale))
            p.setPen(QtGui.QPen(outer_color, pen_w))
            p.drawLine(cx - cs, cy, cx + cs, cy)
            p.drawLine(cx, cy - cs, cx, cy + cs)

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and \
                    self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self.current is not None and len(self.current.points) >= 2):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(int(x), int(y))

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() < w and 0 <= p.y() < h)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        return labelme.utils.distance(p1 - p2) < self.epsilon

    def intersectionPoint(self, p1, p2):
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width() - 1, 0),
                  (size.width() - 1, size.height() - 1),
                  (0, size.height() - 1)]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:
                return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPoint(x2, y2))
                yield d, i, (x, y)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical)
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == QtCore.Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == QtCore.Qt.Key_Return and self.canCloseShape():
            self.finalise()
        # ── 브러시 크기 단축키 ─────────────────────────────────────────
        elif self.isBrushing():
            if key == QtCore.Qt.Key_BracketLeft:    # [ → 작게
                step = 5 if not (ev.modifiers() & QtCore.Qt.ShiftModifier) else 20
                self.brushSize = max(1, self._brushSize - step)
                self.brushSizeChanged.emit(self._brushSize)
                self.update()
            elif key == QtCore.Qt.Key_BracketRight:  # ] → 크게
                step = 5 if not (ev.modifiers() & QtCore.Qt.ShiftModifier) else 20
                self.brushSize = min(500, self._brushSize + step)
                self.brushSizeChanged.emit(self._brushSize)
                self.update()

    def setLastLabel(self, text):
        assert text
        self.shapes[-1].label = text
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ['polygon', 'linestrip']:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ['rectangle', 'line', 'circle']:
            self.current.points = self.current.points[0:1]
        elif self.createMode == 'point':
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.repaint()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.maskImage = None   # 이미지 바뀌면 마스크 초기화
        self.repaint()

    def loadShapes(self, shapes):
        self.shapes = list(shapes)
        self.storeShapes()
        self.current = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.maskImage = None
        self.shapesBackups = []
        self.update()
