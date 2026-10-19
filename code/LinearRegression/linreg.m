// Linear Regression Demo — learning the weight w of  y = w·x + b  by gradient descent
// OpenGL fixed-function pipeline (glBegin/glEnd, glOrtho, glLineStipple) + Cocoa.
//
//  Compile: clang -fobjc-arc -Wno-deprecated-declarations -framework Cocoa -framework OpenGL -o LinRegDemo linreg.m
//  Run:     ./LinRegDemo
//
// Left half : loss curve J(w) with the current w sliding downhill, and w vs. iteration.
// Right half: the 10 data points, the line y = w·x + b being fitted, and error bars
//             drawn as normals (perpendiculars) from each point to the line.
// Stage 1 learns w with b = 2; stage 2 restarts with b = 0, so the line pivots through the origin.
//
// Keys: Space pause · S single step · R restart demo · B toggle b (2 <-> 0)
//       E show/hide error bars · N normals / vertical residuals · L cycle learning rate · +/- animation speed

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>
#include <math.h>

// ============================================================
#pragma mark - Data & model
// ============================================================

#define NPTS 10
#define MAXH 512

static const double kX[NPTS]     = { 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
static const double kNoise[NPTS] = { 0.35, -0.42, 0.18, 0.55, -0.30, -0.62, 0.40, -0.15, 0.48, -0.25 };
static double gY[NPTS];   // generated from y = 1.5x + 2 + noise

static const double kLRs[] = { 0.005, 0.02, 0.06, 0.18 };
#define NLR ((int)(sizeof(kLRs) / sizeof(kLRs[0])))

#define W_MIN  -1.0
#define W_MAX   4.0
#define W_INIT -0.5

static void initData(void) {
    for (int i = 0; i < NPTS; i++) gY[i] = 1.5 * kX[i] + 2.0 + kNoise[i];
}

// J(w) = 1/(2n) Σ (w·xᵢ + b − yᵢ)²
static double lossJ(double w, double b) {
    double s = 0;
    for (int i = 0; i < NPTS; i++) { double r = w * kX[i] + b - gY[i]; s += r * r; }
    return s / (2.0 * NPTS);
}

// dJ/dw = 1/n Σ (w·xᵢ + b − yᵢ)·xᵢ
static double gradJ(double w, double b) {
    double s = 0;
    for (int i = 0; i < NPTS; i++) { double r = w * kX[i] + b - gY[i]; s += r * kX[i]; }
    return s / NPTS;
}

// Closed-form optimum for fixed b: w* = Σ (yᵢ − b)·xᵢ / Σ xᵢ²
static double optimalW(double b) {
    double num = 0, den = 0;
    for (int i = 0; i < NPTS; i++) { num += (gY[i] - b) * kX[i]; den += kX[i] * kX[i]; }
    return num / den;
}

// Foot of the perpendicular from (px,py) to the line y = w·x + b
static void perpFoot(double w, double b, double px, double py, double *fx, double *fy) {
    double inv = 1.0 / sqrt(1.0 + w * w);
    double dx = inv, dy = w * inv;              // unit direction of the line
    double t = px * dx + (py - b) * dy;         // projection of (P − (0,b)) onto direction
    *fx = t * dx;
    *fy = b + t * dy;
}

static double niceStep(double range, int target) {
    double raw = range / target;
    double mag = pow(10, floor(log10(raw)));
    double f = raw / mag;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
}

// ============================================================
#pragma mark - Plot helpers (fixed function)
// ============================================================

// A plot is a pixel rectangle mapped to a data window via glViewport + glOrtho.
typedef struct { float l, b, r, t; double x0, x1, y0, y1; } Plot;

static float PX(const Plot *p, double x) { return p->l + (float)((x - p->x0) / (p->x1 - p->x0)) * (p->r - p->l); }
static float PY(const Plot *p, double y) { return p->b + (float)((y - p->y0) / (p->y1 - p->y0)) * (p->t - p->b); }

static void beginPlot(const Plot *p) {
    GLint x = (GLint)p->l, y = (GLint)p->b;
    GLsizei w = (GLsizei)(p->r - p->l), h = (GLsizei)(p->t - p->b);
    glViewport(x, y, w, h);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x, y, w, h);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(p->x0, p->x1, p->y0, p->y1, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
}

static void drawGrid(const Plot *p, double sx, double sy) {
    glLineWidth(1);
    glColor4f(1, 1, 1, 0.07f);
    glBegin(GL_LINES);
    for (double x = ceil(p->x0 / sx) * sx; x <= p->x1; x += sx) { glVertex2d(x, p->y0); glVertex2d(x, p->y1); }
    for (double y = ceil(p->y0 / sy) * sy; y <= p->y1; y += sy) { glVertex2d(p->x0, y); glVertex2d(p->x1, y); }
    glEnd();
}

static void disc(float cx, float cy, float r) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);
    for (int k = 0; k <= 32; k++) { float a = k * 2.0f * (float)M_PI / 32; glVertex2f(cx + r * cosf(a), cy + r * sinf(a)); }
    glEnd();
}

static void ring(float cx, float cy, float r) {
    glBegin(GL_LINE_LOOP);
    for (int k = 0; k < 32; k++) { float a = k * 2.0f * (float)M_PI / 32; glVertex2f(cx + r * cosf(a), cy + r * sinf(a)); }
    glEnd();
}

static void frameRect(float l, float b, float r, float t) {
    glBegin(GL_LINE_LOOP); glVertex2f(l, b); glVertex2f(r, b); glVertex2f(r, t); glVertex2f(l, t); glEnd();
}

static void fillRect(float l, float b, float r, float t) {
    glBegin(GL_QUADS); glVertex2f(l, b); glVertex2f(r, b); glVertex2f(r, t); glVertex2f(l, t); glEnd();
}

static void setColor(const float *c, float a) { glColor4f(c[0], c[1], c[2], a); }

// Palette
static const float C_TEXT[]   = { 0.90f, 0.92f, 0.96f };
static const float C_DIM[]    = { 0.58f, 0.62f, 0.70f };
static const float C_TITLE[]  = { 1.00f, 1.00f, 1.00f };
static const float C_CURVE[]  = { 0.30f, 0.78f, 1.00f };
static const float C_W[]      = { 1.00f, 0.62f, 0.20f };
static const float C_OPT[]    = { 0.40f, 0.90f, 0.50f };
static const float C_TANGENT[]= { 1.00f, 0.90f, 0.35f };
static const float C_DATA[]   = { 0.55f, 0.75f, 1.00f };
static const float C_ERR[]    = { 1.00f, 0.35f, 0.40f };
static const float C_AXIS[]   = { 0.80f, 0.82f, 0.88f };

// Text alignment flags
enum { AL_LEFT = 0, AL_CENTER = 1, AL_RIGHT = 2, AL_VCENTER = 4 };

// ============================================================
#pragma mark - Text (Cocoa-rendered strings cached as GL textures)
// ============================================================

@interface TexEntry : NSObject { @public GLuint tex; int w, h; }
@end
@implementation TexEntry
@end

// ============================================================
#pragma mark - RegressionView
// ============================================================

enum { PH_LEARN, PH_HOLD };

@interface RegressionView : NSOpenGLView
@end

@implementation RegressionView {
    NSTimer *_timer;
    NSMutableDictionary<NSString *, TexEntry *> *_texCache;
    double _lastTime;

    // model / learning state
    double _w, _prevW, _b, _jMax;
    int _iter, _stage, _phase, _lrIndex;
    double _wHist[MAXH], _jHist[MAXH];
    int _histN;

    // animation state
    double _accum, _stepInterval, _speed, _holdT, _clock;
    BOOL _paused, _showErrors, _showNormals, _autoAdvance;

    // framebuffer
    float _scale;
    int _W, _H;
}

- (instancetype)initWithFrame:(NSRect)frame {
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersionLegacy,
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFAColorSize, 24, NSOpenGLPFAAlphaSize, 8,
        NSOpenGLPFAMultisample, NSOpenGLPFASampleBuffers, 1, NSOpenGLPFASamples, 4,
        0
    };
    NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    if (!pf) {
        NSOpenGLPixelFormatAttribute fallback[] = {
            NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersionLegacy, NSOpenGLPFADoubleBuffer, 0
        };
        pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:fallback];
    }
    if ((self = [super initWithFrame:frame pixelFormat:pf])) {
        [self setWantsBestResolutionOpenGLSurface:YES];
        _texCache = [NSMutableDictionary dictionary];
        _stepInterval = 0.25;
        _speed = 1.0;
        _lrIndex = 1;
        _showErrors = YES;
        _showNormals = YES;
        initData();
        [self restartDemo];
    }
    return self;
}

- (void)prepareOpenGL {
    [super prepareOpenGL];
    GLint swap = 1;
    [[self openGLContext] setValues:&swap forParameter:NSOpenGLCPSwapInterval];
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    _lastTime = CFAbsoluteTimeGetCurrent();
    _timer = [NSTimer timerWithTimeInterval:1.0 / 60 target:self selector:@selector(tick:) userInfo:nil repeats:YES];
    [[NSRunLoop currentRunLoop] addTimer:_timer forMode:NSRunLoopCommonModes];
}

- (BOOL)acceptsFirstResponder { return YES; }

// ------------------------------------------------------------
#pragma mark Learning
// ------------------------------------------------------------

- (double)lr { return kLRs[_lrIndex]; }

- (void)restartDemo {
    _autoAdvance = YES;
    [self startStage:0];
}

- (void)startStage:(int)stage {
    _stage = stage;
    _b = (stage == 0) ? 2.0 : 0.0;
    _w = _prevW = W_INIT;
    _iter = 0;
    _histN = 0;
    _accum = 0;
    _holdT = 0;
    _phase = PH_LEARN;
    [self record];

    _jMax = 0;
    for (int k = 0; k <= 100; k++) _jMax = fmax(_jMax, lossJ(W_MIN + (W_MAX - W_MIN) * k / 100.0, _b));
    _jMax *= 1.08;
}

- (void)record {
    if (_histN < MAXH) {
        _wHist[_histN] = _w;
        _jHist[_histN] = lossJ(_w, _b);
        _histN++;
    }
}

// One gradient-descent update:  w ← w − η · dJ/dw
- (void)gdStep {
    double g = gradJ(_w, _b);
    _prevW = _w;
    _w -= [self lr] * g;
    _iter++;
    [self record];
    if (fabs(_w - _prevW) < 5e-4 || _histN >= MAXH) {
        _phase = PH_HOLD;
        _prevW = _w;
        _holdT = 0;
    }
}

- (void)tick:(NSTimer *)t {
    double now = CFAbsoluteTimeGetCurrent();
    double dt = fmin(now - _lastTime, 0.1);
    _lastTime = now;
    _clock += dt;

    if (!_paused) {
        if (_phase == PH_LEARN) {
            _accum += dt * _speed;
            while (_accum >= _stepInterval && _phase == PH_LEARN) {
                _accum -= _stepInterval;
                [self gdStep];
            }
        } else {
            _holdT += dt;
            if (_autoAdvance && _stage == 0 && _holdT > 3.0) [self startStage:1];
        }
    }
    [self setNeedsDisplay:YES];
}

// Fraction of the way from the previous w to the new w (smoothstep), used to animate between steps
- (double)animT {
    if (_phase != PH_LEARN || _histN < 2) return 1.0;
    double t = fmin(1.0, _accum / _stepInterval);
    return t * t * (3 - 2 * t);
}

- (double)displayW { return _prevW + (_w - _prevW) * [self animT]; }

// ------------------------------------------------------------
#pragma mark Input
// ------------------------------------------------------------

- (void)keyDown:(NSEvent *)e {
    NSString *c = [[e charactersIgnoringModifiers] lowercaseString];
    if ([c isEqualToString:@" "]) {
        _paused = !_paused;
    } else if ([c isEqualToString:@"s"]) {
        _paused = YES;
        if (_phase == PH_LEARN) { _accum = _stepInterval; [self gdStep]; _accum = _stepInterval; }
    } else if ([c isEqualToString:@"r"]) {
        _paused = NO;
        [self restartDemo];
    } else if ([c isEqualToString:@"b"]) {
        _autoAdvance = NO;
        [self startStage:_stage == 0 ? 1 : 0];
    } else if ([c isEqualToString:@"e"]) {
        _showErrors = !_showErrors;
    } else if ([c isEqualToString:@"n"]) {
        _showNormals = !_showNormals;
    } else if ([c isEqualToString:@"l"]) {
        _lrIndex = (_lrIndex + 1) % NLR;
        [self startStage:_stage];
    } else if ([c isEqualToString:@"+"] || [c isEqualToString:@"="]) {
        _speed = fmin(_speed * 1.5, 16.0);
    } else if ([c isEqualToString:@"-"]) {
        _speed = fmax(_speed / 1.5, 0.125);
    } else if ([c isEqualToString:@"q"] || [c isEqualToString:@"\e"]) {
        [NSApp terminate:nil];
    } else {
        [super keyDown:e];
    }
}

// ------------------------------------------------------------
#pragma mark Text rendering
// ------------------------------------------------------------

- (TexEntry *)texFor:(NSString *)s size:(float)pt {
    NSString *key = [NSString stringWithFormat:@"%.1f|%.1f|%@", pt, _scale, s];
    TexEntry *e = _texCache[key];
    if (e) return e;

    if (_texCache.count > 400) {
        for (TexEntry *old in _texCache.allValues) glDeleteTextures(1, &old->tex);
        [_texCache removeAllObjects];
    }

    NSFont *font = [NSFont fontWithName:@"Menlo" size:pt * _scale];
    if (!font) font = [NSFont monospacedSystemFontOfSize:pt * _scale weight:NSFontWeightRegular];
    NSDictionary *attr = @{ NSFontAttributeName: font, NSForegroundColorAttributeName: [NSColor whiteColor] };
    NSSize sz = [s sizeWithAttributes:attr];
    int w = (int)ceil(sz.width) + 2, h = (int)ceil(sz.height) + 2;

    NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL pixelsWide:w pixelsHigh:h bitsPerSample:8 samplesPerPixel:4
                        hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace
                     bytesPerRow:w * 4 bitsPerPixel:32];
    memset([rep bitmapData], 0, (size_t)w * h * 4);

    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithBitmapImageRep:rep]];
    [s drawAtPoint:NSMakePoint(1, 1) withAttributes:attr];
    [[NSGraphicsContext currentContext] flushGraphics];
    [NSGraphicsContext restoreGraphicsState];

    e = [TexEntry new];
    e->w = w; e->h = h;
    glGenTextures(1, &e->tex);
    glBindTexture(GL_TEXTURE_2D, e->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, [rep bitmapData]);
    _texCache[key] = e;
    return e;
}

// Draws text in window pixel coordinates; (x, y) is the bottom-left unless align says otherwise.
- (void)text:(NSString *)s x:(float)x y:(float)y size:(float)pt color:(const float *)c align:(int)align {
    TexEntry *e = [self texFor:s size:pt];
    if ((align & 3) == AL_CENTER) x -= e->w * 0.5f;
    else if ((align & 3) == AL_RIGHT) x -= e->w;
    if (align & AL_VCENTER) y -= e->h * 0.5f;
    x = floorf(x); y = floorf(y);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, e->tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);     // bitmap is premultiplied
    glColor4f(c[0], c[1], c[2], 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(x, y);
    glTexCoord2f(1, 1); glVertex2f(x + e->w, y);
    glTexCoord2f(1, 0); glVertex2f(x + e->w, y + e->h);
    glTexCoord2f(0, 0); glVertex2f(x, y + e->h);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// ------------------------------------------------------------
#pragma mark Rendering
// ------------------------------------------------------------

- (void)pixelSpace {
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, _W, _H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(0, _W, 0, _H, -1, 1);
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
}

- (void)axisLabelsFor:(const Plot *)p sx:(double)sx sy:(double)sy fmtX:(NSString *)fx fmtY:(NSString *)fy {
    float s = _scale;
    for (double x = ceil(p->x0 / sx - 1e-9) * sx; x <= p->x1 + 1e-9; x += sx) {
        double v = fabs(x) < 1e-9 ? 0 : x;
        [self text:[NSString stringWithFormat:fx, v] x:PX(p, x) y:p->b - 18 * s size:10 color:C_DIM align:AL_CENTER];
    }
    for (double y = ceil(p->y0 / sy - 1e-9) * sy; y <= p->y1 + 1e-9; y += sy) {
        double v = fabs(y) < 1e-9 ? 0 : y;
        [self text:[NSString stringWithFormat:fy, v] x:p->l - 6 * s y:PY(p, y) size:10 color:C_DIM align:AL_RIGHT | AL_VCENTER];
    }
    glLineWidth(1);
    glColor4f(1, 1, 1, 0.25f);
    frameRect(p->l, p->b, p->r, p->t);
}

- (void)drawRect:(NSRect)dirty {
    [[self openGLContext] makeCurrentContext];
    NSRect bb = [self convertRectToBacking:[self bounds]];
    _W = (int)bb.size.width;
    _H = (int)bb.size.height;
    _scale = (float)(bb.size.width / fmax(1.0, self.bounds.size.width));

    glClearColor(0.07f, 0.08f, 0.10f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float mid = floorf(_W * 0.5f);
    [self pixelSpace];
    glColor4f(1, 1, 1, 0.025f);
    fillRect(0, 0, mid, _H);
    glColor4f(1, 1, 1, 0.15f);
    glLineWidth(1);
    glBegin(GL_LINES); glVertex2f(mid, 0); glVertex2f(mid, _H); glEnd();

    [self drawLeftPanel:mid];
    [self drawRightPanel:mid];

    [[self openGLContext] flushBuffer];
}

// ---------------- Left: how w is learned ----------------

- (void)drawLeftPanel:(float)mid {
    float s = _scale;
    double dw = [self displayW], at = [self animT];
    double wStar = optimalW(_b);
    int shown = (_phase == PH_LEARN && _histN > 1) ? _histN - 1 : _histN;   // history points already reached

    [self text:@"LEFT  ·  Learning the weight w by gradient descent" x:20 * s y:_H - 32 * s size:15 color:C_TITLE align:AL_LEFT];
    [self text:@"J(w) = 1/(2n) Σ (w·xᵢ + b − yᵢ)²        update:  w ← w − η · dJ/dw"
             x:20 * s y:_H - 54 * s size:12 color:C_DIM align:AL_LEFT];

    // ---- Loss curve J(w) ----
    float split = floorf(_H * 0.46f);
    Plot lp = { floorf(70 * s), split, floorf(mid - 30 * s), floorf(_H - 90 * s), W_MIN, W_MAX, 0, _jMax };
    double syL = niceStep(_jMax, 5);

    beginPlot(&lp);
    drawGrid(&lp, 1.0, syL);

    // w* marker
    glEnable(GL_LINE_STIPPLE); glLineStipple((GLint)(2 * s), 0xAAAA);
    glLineWidth(1.5f * s); setColor(C_OPT, 0.8f);
    glBegin(GL_LINES); glVertex2d(wStar, 0); glVertex2d(wStar, _jMax); glEnd();
    glDisable(GL_LINE_STIPPLE);

    // the parabola J(w)
    glLineWidth(2.5f * s); setColor(C_CURVE, 1);
    glBegin(GL_LINE_STRIP);
    for (int k = 0; k <= 200; k++) { double w = W_MIN + (W_MAX - W_MIN) * k / 200.0; glVertex2d(w, lossJ(w, _b)); }
    glEnd();

    // path taken by gradient descent
    glLineWidth(1.5f * s); setColor(C_W, 0.7f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < shown; i++) glVertex2d(_wHist[i], _jHist[i]);
    glVertex2d(dw, lossJ(dw, _b));
    glEnd();

    // tangent (slope = dJ/dw) at the current w
    double g = gradJ(dw, _b), jw = lossJ(dw, _b);
    glLineWidth(2 * s); setColor(C_TANGENT, 0.9f);
    glBegin(GL_LINES); glVertex2d(dw - 0.55, jw - 0.55 * g); glVertex2d(dw + 0.55, jw + 0.55 * g); glEnd();
    [self pixelSpace];

    setColor(C_W, 0.8f);
    for (int i = 0; i < shown; i++) disc(PX(&lp, _wHist[i]), PY(&lp, _jHist[i]), 2.5f * s);
    setColor(C_W, 1); disc(PX(&lp, dw), PY(&lp, jw), 7 * s);
    glLineWidth(1.5f * s); glColor4f(1, 1, 1, 0.9f); ring(PX(&lp, dw), PY(&lp, jw), 7 * s);

    [self axisLabelsFor:&lp sx:1.0 sy:syL fmtX:@"%.0f" fmtY:syL < 1 ? @"%.1f" : @"%.0f"];
    [self text:@"w" x:(lp.l + lp.r) * 0.5f y:lp.b - 36 * s size:12 color:C_TEXT align:AL_CENTER];
    [self text:@"J(w)" x:lp.l + 6 * s y:lp.t - 20 * s size:12 color:C_CURVE align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"w* = %.3f", wStar] x:PX(&lp, wStar) + 6 * s y:lp.t - 20 * s
          size:11 color:C_OPT align:AL_LEFT];

    // live numbers, top-centre (above the parabola's valley)
    float ix = PX(&lp, 0.35), iy = lp.t - 50 * s, lh = 17 * s;
    [self text:[NSString stringWithFormat:@"iteration  %d", _iter] x:ix y:iy size:12 color:C_TEXT align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"w          %+.4f", dw] x:ix y:iy - lh size:12 color:C_W align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"dJ/dw      %+.4f", g] x:ix y:iy - 2 * lh size:12 color:C_TANGENT align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"J(w)       %.4f", jw] x:ix y:iy - 3 * lh size:12 color:C_CURVE align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"η (lr)     %.3f", [self lr]] x:ix y:iy - 4 * lh size:12 color:C_DIM align:AL_LEFT];

    // ---- w vs iteration ----
    int itMax = MAX(40, ((_histN + 9) / 10) * 10);
    Plot ip = { floorf(70 * s), floorf(60 * s), floorf(mid - 30 * s), floorf(split - 70 * s), 0, itMax, -1, 4 };
    double sxI = niceStep(itMax, 8);

    beginPlot(&ip);
    drawGrid(&ip, sxI, 1.0);
    glEnable(GL_LINE_STIPPLE); glLineStipple((GLint)(2 * s), 0xAAAA);
    glLineWidth(1.5f * s); setColor(C_OPT, 0.8f);
    glBegin(GL_LINES); glVertex2d(0, wStar); glVertex2d(itMax, wStar); glEnd();
    glDisable(GL_LINE_STIPPLE);

    glLineWidth(2 * s); setColor(C_W, 1);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < shown; i++) glVertex2d(i, _wHist[i]);
    glVertex2d(shown - 1 + (_phase == PH_LEARN && _histN > 1 ? at : 0), dw);
    glEnd();
    [self pixelSpace];

    setColor(C_W, 0.9f);
    for (int i = 0; i < shown; i++) disc(PX(&ip, i), PY(&ip, _wHist[i]), 2.5f * s);

    [self axisLabelsFor:&ip sx:sxI sy:1.0 fmtX:@"%.0f" fmtY:@"%.0f"];
    [self text:@"iteration" x:(ip.l + ip.r) * 0.5f y:ip.b - 36 * s size:12 color:C_TEXT align:AL_CENTER];
    [self text:@"w  vs  iteration" x:ip.l + 6 * s y:ip.t - 20 * s size:12 color:C_W align:AL_LEFT];
    [self text:[NSString stringWithFormat:@"w* = %.3f  (closed-form optimum for b = %.0f)", wStar, _b]
             x:ip.r - 8 * s y:PY(&ip, wStar) + 5 * s size:11 color:C_OPT align:AL_RIGHT];
}

// ---------------- Right: the fitted line ----------------

- (void)drawRightPanel:(float)mid {
    float s = _scale;
    double dw = [self displayW], b = _b;

    [self text:@"RIGHT  ·  Fitting the line  y = w·x + b" x:mid + 20 * s y:_H - 32 * s size:15 color:C_TITLE align:AL_LEFT];
    NSString *stageMsg = (_stage == 0)
        ? @"Stage 1 of 2:  b = 2 (held fixed) — only w is learned"
        : @"Stage 2 of 2:  b = 0 — the line is forced through the origin (0, 0)";
    [self text:stageMsg x:mid + 20 * s y:_H - 54 * s size:12 color:(_stage == 0 ? C_TEXT : C_OPT) align:AL_LEFT];

    // Equal scale on both axes so the normals really look perpendicular.
    float l = floorf(mid + 60 * s), r = floorf(_W - 30 * s), bt = floorf(80 * s), t = floorf(_H - 120 * s);
    double sc = fmin((r - l) / 7.0, (t - bt) / 12.0);                 // pixels per data unit
    double hx = (r - l) / sc * 0.5, hy = (t - bt) / sc * 0.5;
    Plot rp = { l, bt, r, t, 2.5 - hx, 2.5 + hx, 4.5 - hy, 4.5 + hy };
    double gs = sc < 30 ? 2.0 : 1.0;

    // error statistics
    double sumPerp = 0, sse = 0;
    for (int i = 0; i < NPTS; i++) {
        double res = dw * kX[i] + b - gY[i];
        sse += res * res;
        sumPerp += fabs(res) / sqrt(1 + dw * dw);
    }

    beginPlot(&rp);
    drawGrid(&rp, gs, gs);

    // x and y axes through the origin
    glLineWidth(1.5f * s); setColor(C_AXIS, 0.6f);
    glBegin(GL_LINES);
    glVertex2d(rp.x0, 0); glVertex2d(rp.x1, 0);
    glVertex2d(0, rp.y0); glVertex2d(0, rp.y1);
    glEnd();

    // error bars: normals (perpendicular) or vertical residuals
    if (_showErrors) {
        glLineWidth(3 * s); setColor(C_ERR, 0.95f);
        glBegin(GL_LINES);
        for (int i = 0; i < NPTS; i++) {
            double fx = kX[i], fy = dw * kX[i] + b;
            if (_showNormals) perpFoot(dw, b, kX[i], gY[i], &fx, &fy);
            glVertex2d(kX[i], gY[i]); glVertex2d(fx, fy);
        }
        glEnd();
    }

    // right-angle markers where each normal meets the line
    if (_showErrors && _showNormals) {
        double inv = 1.0 / sqrt(1 + dw * dw), dx = inv, dy = dw * inv, a = 0.14;
        glLineWidth(1.2f * s); setColor(C_ERR, 0.7f);
        for (int i = 0; i < NPTS; i++) {
            double fx, fy;
            perpFoot(dw, b, kX[i], gY[i], &fx, &fy);
            double nx = kX[i] - fx, ny = gY[i] - fy, len = sqrt(nx * nx + ny * ny);
            if (len < 2.5 * a) continue;
            nx /= len; ny /= len;
            glBegin(GL_LINE_STRIP);
            glVertex2d(fx + a * nx, fy + a * ny);
            glVertex2d(fx + a * nx + a * dx, fy + a * ny + a * dy);
            glVertex2d(fx + a * dx, fy + a * dy);
            glEnd();
        }
    }

    // the fitted line
    glLineWidth(3 * s); setColor(C_W, 1);
    glBegin(GL_LINES); glVertex2d(rp.x0, dw * rp.x0 + b); glVertex2d(rp.x1, dw * rp.x1 + b); glEnd();
    [self pixelSpace];

    // origin marker (pulses in stage 2)
    float ox = PX(&rp, 0), oy = PY(&rp, 0);
    float pulse = (_stage == 1) ? 1.0f + 0.25f * sinf((float)_clock * 5) : 1.0f;
    glLineWidth(2 * s); setColor(_stage == 1 ? C_OPT : C_AXIS, _stage == 1 ? 1.0f : 0.5f);
    ring(ox, oy, 9 * s * pulse);
    [self text:@"(0, 0)" x:ox - 8 * s y:oy - 22 * s size:11 color:(_stage == 1 ? C_OPT : C_DIM) align:AL_RIGHT];

    // intercept marker at (0, b)
    if (b != 0) {
        setColor(C_W, 1); disc(ox, PY(&rp, b), 4.5f * s);
        [self text:[NSString stringWithFormat:@"b = %.0f", b] x:ox - 8 * s y:PY(&rp, b) size:11 color:C_W align:AL_RIGHT | AL_VCENTER];
    }

    // feet of the error bars + data points
    for (int i = 0; _showErrors && i < NPTS; i++) {
        double fx = kX[i], fy = dw * kX[i] + b;
        if (_showNormals) perpFoot(dw, b, kX[i], gY[i], &fx, &fy);
        setColor(C_ERR, 1); disc(PX(&rp, fx), PY(&rp, fy), 2.5f * s);
    }
    for (int i = 0; i < NPTS; i++) {
        float px = PX(&rp, kX[i]), py = PY(&rp, gY[i]);
        setColor(C_DATA, 1); disc(px, py, 5 * s);
        glLineWidth(1.5f * s); glColor4f(1, 1, 1, 0.9f); ring(px, py, 5 * s);
    }

    // line equation label, placed on the upper-left side of the line (away from the data)
    double ly = rp.y0 + 0.88 * (rp.y1 - rp.y0);
    double lx = fabs(dw) > 0.05 ? (ly - b) / dw : 4.0;
    lx = fmin(fmax(lx, 0.5), rp.x1 - 1.0);
    [self text:[NSString stringWithFormat:@"y = %.3f·x %@ %.0f", dw, b < 0 ? @"−" : @"+", fabs(b)]
             x:PX(&rp, lx) - 30 * s y:PY(&rp, dw * lx + b) + 10 * s size:12 color:C_W align:AL_RIGHT | AL_VCENTER];

    [self axisLabelsFor:&rp sx:gs sy:gs fmtX:@"%.0f" fmtY:@"%.0f"];
    [self text:@"x" x:rp.r - 10 * s y:oy + 6 * s size:12 color:C_TEXT align:AL_CENTER];
    [self text:@"y" x:ox + 10 * s y:rp.t - 20 * s size:12 color:C_TEXT align:AL_CENTER];

    // HUD
    float hy0 = _H - 80 * s;
    [self text:[NSString stringWithFormat:@"w = %+.4f    b = %.2f    J = %.4f    SSE = %.3f    mean ⊥ dist = %.3f",
                dw, b, lossJ(dw, b), sse, sumPerp / NPTS]
             x:mid + 20 * s y:hy0 size:12 color:C_TEXT align:AL_LEFT];

    NSString *status;
    if (_paused) status = @"⏸  PAUSED  (Space to resume, S to step)";
    else if (_phase == PH_LEARN) status = [NSString stringWithFormat:@"learning…   speed ×%.2g", _speed];
    else if (_autoAdvance && _stage == 0) status = [NSString stringWithFormat:@"✓ converged — switching to b = 0 in %.1f s", fmax(0, 3.0 - _holdT)];
    else status = @"✓ converged — press R to replay, B to toggle b";
    [self text:status x:mid + 20 * s y:hy0 - 20 * s size:12 color:(_phase == PH_LEARN ? C_TANGENT : C_OPT) align:AL_LEFT];

    // legend + key help
    NSString *errName = !_showErrors ? @"error bars hidden (E to show)"
                      : _showNormals ? @"normals ⊥ to line (error bars)" : @"vertical residuals yᵢ − ŷᵢ (what least squares minimises)";
    [self text:[NSString stringWithFormat:@"●  data (xᵢ, yᵢ)      —  fitted line      |  %@", errName]
             x:mid + 20 * s y:36 * s size:11 color:C_DIM align:AL_LEFT];
    [self text:@"[Space] pause  [S] step  [R] restart  [B] toggle b  [E] error bars  [N] normals/vertical  [L] lr  [+/−] speed"
             x:mid + 20 * s y:14 * s size:10 color:C_DIM align:AL_LEFT];
}

@end

// ============================================================
#pragma mark - App
// ============================================================

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (strong) NSWindow *window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)n {
    NSMenu *bar = [NSMenu new];
    NSMenuItem *appItem = [NSMenuItem new];
    [bar addItem:appItem];
    NSMenu *appMenu = [NSMenu new];
    [appMenu addItemWithTitle:@"Quit Linear Regression Demo" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    NSApp.mainMenu = bar;

    NSRect frame = NSMakeRect(0, 0, 1400, 820);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                        NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                                                backing:NSBackingStoreBuffered defer:NO];
    self.window.title = @"Linear Regression — learning w in y = w·x + b (OpenGL fixed function)";
    self.window.contentMinSize = NSMakeSize(1000, 640);
    RegressionView *view = [[RegressionView alloc] initWithFrame:frame];
    self.window.contentView = view;
    [self.window center];
    [self.window makeKeyAndOrderFront:nil];
    [self.window makeFirstResponder:view];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app { return YES; }

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate *delegate = [AppDelegate new];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
