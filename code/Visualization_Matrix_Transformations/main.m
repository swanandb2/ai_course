// 3D Matrix Transformation Demo
// Compile: clang -fobjc-arc -Wno-deprecated-declarations -framework Cocoa -framework OpenGL -o MatrixDemo main.m
// Run:     ./MatrixDemo

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>
#import <OpenGL/glu.h>
#include <math.h>

// ============================================================
#pragma mark - Vec3
// ============================================================

typedef struct { float x, y, z; } Vec3;

static inline Vec3 v3(float x, float y, float z) { return (Vec3){x, y, z}; }
static inline Vec3 v3Add(Vec3 a, Vec3 b)   { return (Vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3 v3Sub(Vec3 a, Vec3 b)   { return (Vec3){a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3 v3Scale(Vec3 v, float s) { return (Vec3){v.x*s, v.y*s, v.z*s}; }
static inline float v3Len(Vec3 v)           { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }

static inline Vec3 v3Norm(Vec3 v) {
    float l = v3Len(v);
    return (l < 1e-6f) ? v3(0,0,0) : v3Scale(v, 1.0f/l);
}

static inline Vec3 v3Cross(Vec3 a, Vec3 b) {
    return (Vec3){ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

// ============================================================
#pragma mark - Mat3 (m[row][col])
// ============================================================

typedef struct { float m[3][3]; } Mat3;

static Vec3 m3MulV(Mat3 M, Vec3 v) {
    return (Vec3){
        M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z,
        M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z,
        M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z
    };
}

static float m3Det(Mat3 M) {
    return M.m[0][0]*(M.m[1][1]*M.m[2][2] - M.m[1][2]*M.m[2][1])
         - M.m[0][1]*(M.m[1][0]*M.m[2][2] - M.m[1][2]*M.m[2][0])
         + M.m[0][2]*(M.m[1][0]*M.m[2][1] - M.m[1][1]*M.m[2][0]);
}

static Vec3 m3Col(Mat3 M, int j) {
    return (Vec3){ M.m[0][j], M.m[1][j], M.m[2][j] };
}

static Mat3 m3Id(void) {
    return (Mat3){{{1,0,0},{0,1,0},{0,0,1}}};
}

static Mat3 m3Lerp(Mat3 A, Mat3 B, float t) {
    Mat3 R;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R.m[i][j] = A.m[i][j] + t * (B.m[i][j] - A.m[i][j]);
    return R;
}

static float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

// ============================================================
#pragma mark - Predefined Matrices
// ============================================================

static Mat3 matScale(void) {
    return (Mat3){{{2.0f, 0, 0}, {0, 1.5f, 0}, {0, 0, 1.0f}}};
}

static Mat3 matRotZ(float angle) {
    float c = cosf(angle), s = sinf(angle);
    return (Mat3){{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

static Mat3 matShear(void) {
    return (Mat3){{{1, 0.8f, 0}, {0, 1, 0.6f}, {0, 0, 1}}};
}

// ============================================================
#pragma mark - GLView
// ============================================================

@interface GLView : NSOpenGLView {
    float _rotX, _rotY, _zoom;
    NSPoint _lastMouse;
    int _mode;        // 1=scale 2=rotate 3=shear
    float _t;         // animation parameter 0..1
    int _tDir;
    BOOL _paused;
    NSTimer *_timer;
    NSTextField *_hud;
    NSTextField *_legend;
}
@end

@implementation GLView

- (instancetype)initWithFrame:(NSRect)frame {
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize, 24,
        NSOpenGLPFAMultisample,
        NSOpenGLPFASampleBuffers, 1,
        NSOpenGLPFASamples, 4,
        0
    };
    NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    self = [super initWithFrame:frame pixelFormat:pf];
    if (self) {
        _rotX = 25; _rotY = -35; _zoom = 7;
        _mode = 1; _t = 0; _tDir = 1; _paused = NO;
        self.wantsBestResolutionOpenGLSurface = YES;
    }
    return self;
}

- (void)prepareOpenGL {
    [super prepareOpenGL];
    GLint swap = 1;
    [[self openGLContext] setValues:&swap forParameter:NSOpenGLCPSwapInterval];

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.07f, 0.07f, 0.11f, 1.0f);

    NSShadow *sh = [[NSShadow alloc] init];
    sh.shadowColor = [NSColor colorWithWhite:0 alpha:0.95];
    sh.shadowOffset = NSMakeSize(1, -1);
    sh.shadowBlurRadius = 4;

    _hud = [NSTextField wrappingLabelWithString:@""];
    _hud.bordered = NO;
    _hud.drawsBackground = NO;
    _hud.textColor = [NSColor colorWithRed:0.88 green:0.93 blue:1.0 alpha:1.0];
    _hud.font = [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightMedium];
    _hud.shadow = sh;
    [self addSubview:_hud];

    _legend = [NSTextField wrappingLabelWithString:
        @"[1] Scale  [2] Rotate  [3] Shear   |   [Space] Pause   [R] Reset   [V] Default view   [Drag] Rotate   [Scroll] Zoom"];
    _legend.bordered = NO;
    _legend.drawsBackground = NO;
    _legend.textColor = [NSColor colorWithWhite:0.6 alpha:1.0];
    _legend.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    _legend.shadow = sh;
    [self addSubview:_legend];

    _timer = [NSTimer scheduledTimerWithTimeInterval:1.0/60 target:self
              selector:@selector(tick:) userInfo:nil repeats:YES];
}

- (BOOL)acceptsFirstResponder { return YES; }

- (void)reshape {
    [super reshape];
    NSRect bp = [self convertRectToBacking:self.bounds];
    [[self openGLContext] makeCurrentContext];
    glViewport(0, 0, (GLint)bp.size.width, (GLint)bp.size.height);

    NSRect b = self.bounds;
    _hud.frame = NSMakeRect(12, b.size.height - 175, 540, 168);
    _legend.frame = NSMakeRect(12, 8, b.size.width - 24, 16);
}

- (void)viewDidMoveToWindow {
    if (!self.window && _timer) { [_timer invalidate]; _timer = nil; }
}

// ============================================================
#pragma mark - Animation
// ============================================================

- (void)tick:(NSTimer *)t {
    if (_paused) return;
    if (_mode == 2) {
        _t += 0.004f;
        if (_t >= 1.0f) _t -= 1.0f;
    } else {
        _t += 0.005f * _tDir;
        if (_t >= 1.0f) { _t = 1.0f; _tDir = -1; }
        if (_t <= 0.0f) { _t = 0.0f; _tDir = 1; }
    }
    [self setNeedsDisplay:YES];
}

- (Mat3)currentMatrix {
    if (_mode == 2) return matRotZ(_t * 2.0f * M_PI);
    Mat3 target = (_mode == 3) ? matShear() : matScale();
    return m3Lerp(m3Id(), target, smoothstep(_t));
}

// ============================================================
#pragma mark - Drawing Helpers
// ============================================================

- (void)drawConeAt:(Vec3)tip dir:(Vec3)dir r:(float)cr g:(float)cg b:(float)cb
            radius:(float)rad length:(float)len {
    float dl = v3Len(dir);
    if (dl < 1e-4f) return;
    Vec3 d = v3Scale(dir, 1.0f/dl);
    Vec3 base = v3Add(tip, v3Scale(d, -len));
    Vec3 up = (fabsf(d.y) < 0.9f) ? v3(0,1,0) : v3(1,0,0);
    Vec3 ri = v3Norm(v3Cross(d, up));
    Vec3 fw = v3Cross(ri, d);

    glColor3f(cr, cg, cb);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(tip.x, tip.y, tip.z);
    for (int i = 0; i <= 16; i++) {
        float a = 2.0f * M_PI * i / 16.0f;
        Vec3 p = v3Add(base, v3Add(v3Scale(ri, cosf(a)*rad), v3Scale(fw, sinf(a)*rad)));
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
}

- (void)drawArrowFrom:(Vec3)from to:(Vec3)to r:(float)cr g:(float)cg b:(float)cb width:(float)w {
    Vec3 dir = v3Sub(to, from);
    glLineWidth(w);
    glColor3f(cr, cg, cb);
    glBegin(GL_LINES);
    glVertex3f(from.x, from.y, from.z);
    glVertex3f(to.x, to.y, to.z);
    glEnd();
    [self drawConeAt:to dir:dir r:cr g:cg b:cb radius:0.04f length:0.13f];
}

- (void)drawGrid {
    glLineWidth(1.0f);
    glColor4f(0.22f, 0.22f, 0.28f, 0.5f);
    glBegin(GL_LINES);
    for (float i = -4; i <= 4; i += 0.5f) {
        glVertex3f(i, 0, -4); glVertex3f(i, 0, 4);
        glVertex3f(-4, 0, i); glVertex3f(4, 0, i);
    }
    glEnd();
}

- (void)drawAxes {
    float L = 3.5f;
    glLineWidth(1.5f);

    glColor3f(0.75f, 0.22f, 0.22f);
    glBegin(GL_LINES); glVertex3f(-L,0,0); glVertex3f(L,0,0); glEnd();
    [self drawConeAt:v3(L,0,0) dir:v3(1,0,0) r:0.75 g:0.22 b:0.22 radius:0.05 length:0.15];

    glColor3f(0.22f, 0.75f, 0.22f);
    glBegin(GL_LINES); glVertex3f(0,-L,0); glVertex3f(0,L,0); glEnd();
    [self drawConeAt:v3(0,L,0) dir:v3(0,1,0) r:0.22 g:0.75 b:0.22 radius:0.05 length:0.15];

    glColor3f(0.28f, 0.28f, 0.85f);
    glBegin(GL_LINES); glVertex3f(0,0,-L); glVertex3f(0,0,L); glEnd();
    [self drawConeAt:v3(0,0,L) dir:v3(0,0,1) r:0.28 g:0.28 b:0.85 radius:0.05 length:0.15];

    // Tick marks
    glColor4f(0.5f, 0.5f, 0.5f, 0.4f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int i = -3; i <= 3; i++) {
        if (i == 0) continue;
        float s = 0.05f;
        glVertex3f(i, -s, 0); glVertex3f(i, s, 0);
        glVertex3f(-s, i, 0); glVertex3f(s, i, 0);
        glVertex3f(0, -s, i); glVertex3f(0, s, i);
    }
    glEnd();

    // Origin
    glPointSize(5.0f);
    glColor3f(1, 1, 1);
    glBegin(GL_POINTS); glVertex3f(0, 0, 0); glEnd();
}

- (void)drawParallelepiped:(Mat3)M edgeR:(float)er g:(float)eg b:(float)eb
                 edgeAlpha:(float)ea faceR:(float)fr g:(float)fg b:(float)fb faceAlpha:(float)fa {
    Vec3 a = m3Col(M, 0), b = m3Col(M, 1), c = m3Col(M, 2);
    Vec3 o  = v3(0,0,0);
    Vec3 ab = v3Add(a, b), ac = v3Add(a, c), bc = v3Add(b, c), abc = v3Add(ab, c);

    // Edges
    glLineWidth(1.5f);
    glColor4f(er, eg, eb, ea);
    glBegin(GL_LINES);
    glVertex3f(o.x,o.y,o.z); glVertex3f(a.x,a.y,a.z);
    glVertex3f(o.x,o.y,o.z); glVertex3f(b.x,b.y,b.z);
    glVertex3f(o.x,o.y,o.z); glVertex3f(c.x,c.y,c.z);
    glVertex3f(a.x,a.y,a.z); glVertex3f(ab.x,ab.y,ab.z);
    glVertex3f(a.x,a.y,a.z); glVertex3f(ac.x,ac.y,ac.z);
    glVertex3f(b.x,b.y,b.z); glVertex3f(ab.x,ab.y,ab.z);
    glVertex3f(b.x,b.y,b.z); glVertex3f(bc.x,bc.y,bc.z);
    glVertex3f(c.x,c.y,c.z); glVertex3f(ac.x,ac.y,ac.z);
    glVertex3f(c.x,c.y,c.z); glVertex3f(bc.x,bc.y,bc.z);
    glVertex3f(ab.x,ab.y,ab.z); glVertex3f(abc.x,abc.y,abc.z);
    glVertex3f(ac.x,ac.y,ac.z); glVertex3f(abc.x,abc.y,abc.z);
    glVertex3f(bc.x,bc.y,bc.z); glVertex3f(abc.x,abc.y,abc.z);
    glEnd();

    // Transparent faces
    glDepthMask(GL_FALSE);
    glColor4f(fr, fg, fb, fa);
    glBegin(GL_QUADS);
    // 6 faces
    glVertex3f(o.x,o.y,o.z); glVertex3f(a.x,a.y,a.z);
    glVertex3f(ab.x,ab.y,ab.z); glVertex3f(b.x,b.y,b.z);

    glVertex3f(o.x,o.y,o.z); glVertex3f(a.x,a.y,a.z);
    glVertex3f(ac.x,ac.y,ac.z); glVertex3f(c.x,c.y,c.z);

    glVertex3f(o.x,o.y,o.z); glVertex3f(b.x,b.y,b.z);
    glVertex3f(bc.x,bc.y,bc.z); glVertex3f(c.x,c.y,c.z);

    glVertex3f(abc.x,abc.y,abc.z); glVertex3f(ab.x,ab.y,ab.z);
    glVertex3f(a.x,a.y,a.z); glVertex3f(ac.x,ac.y,ac.z);

    glVertex3f(abc.x,abc.y,abc.z); glVertex3f(ab.x,ab.y,ab.z);
    glVertex3f(b.x,b.y,b.z); glVertex3f(bc.x,bc.y,bc.z);

    glVertex3f(abc.x,abc.y,abc.z); glVertex3f(ac.x,ac.y,ac.z);
    glVertex3f(c.x,c.y,c.z); glVertex3f(bc.x,bc.y,bc.z);
    glEnd();
    glDepthMask(GL_TRUE);
}

// ============================================================
#pragma mark - Main Render
// ============================================================

- (void)drawRect:(NSRect)dirtyRect {
    [[self openGLContext] makeCurrentContext];
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    NSRect bounds = self.bounds;
    float aspect = bounds.size.width / bounds.size.height;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -_zoom);
    glRotatef(_rotX, 1, 0, 0);
    glRotatef(_rotY, 0, 1, 0);

    [self drawGrid];
    [self drawAxes];

    Mat3 M = [self currentMatrix];
    Vec3 v = v3(1.0f, 0.7f, 0.5f);
    Vec3 Mv = m3MulV(M, v);

    // Unit cube (original) — dim white wireframe
    [self drawParallelepiped:m3Id()
        edgeR:0.5 g:0.5 b:0.5 edgeAlpha:0.2
        faceR:0.4 g:0.4 b:0.4 faceAlpha:0.04];

    // Transformed parallelepiped — cyan
    [self drawParallelepiped:M
        edgeR:0.15 g:0.7 b:0.95 edgeAlpha:0.65
        faceR:0.1 g:0.35 b:0.55 faceAlpha:0.1];

    // Transformed basis vectors (colored to match axes)
    Vec3 e1 = m3Col(M, 0), e2 = m3Col(M, 1), e3 = m3Col(M, 2);
    Vec3 o = v3(0,0,0);
    [self drawArrowFrom:o to:e1 r:1.0 g:0.35 b:0.35 width:2.5];
    [self drawArrowFrom:o to:e2 r:0.35 g:1.0 b:0.35 width:2.5];
    [self drawArrowFrom:o to:e3 r:0.45 g:0.45 b:1.0 width:2.5];

    // Original vector — yellow
    [self drawArrowFrom:o to:v r:1.0 g:0.85 b:0.1 width:3.0];

    // Transformed vector — bright cyan
    [self drawArrowFrom:o to:Mv r:0.1 g:1.0 b:0.95 width:3.0];

    [[self openGLContext] flushBuffer];

    [self updateHUD:M v:v Mv:Mv];
}

// ============================================================
#pragma mark - HUD
// ============================================================

- (void)updateHUD:(Mat3)M v:(Vec3)v Mv:(Vec3)Mv {
    float det = m3Det(M);
    NSString *mode, *desc;
    switch (_mode) {
        case 1:  mode = @"SCALING";  desc = @"det = product of scale factors = volume ratio"; break;
        case 2:  mode = @"ROTATION"; desc = @"det = 1 always — lengths & volume preserved"; break;
        default: mode = @"SHEAR";    desc = @"det = 1 — volume preserved, shape changes"; break;
    }

    NSString *text = [NSString stringWithFormat:
        @"%@ — %@\n\n"
        @"     ┌%6.2f %6.2f %6.2f ┐\n"
        @"M =  │%6.2f %6.2f %6.2f │  det(M) = %.3f\n"
        @"     └%6.2f %6.2f %6.2f ┘  Vol  = %.3fx\n\n"
        @"v  = (%5.2f, %5.2f, %5.2f)  |v|  = %.3f\n"
        @"Mv = (%5.2f, %5.2f, %5.2f)  |Mv| = %.3f",
        mode, desc,
        M.m[0][0], M.m[0][1], M.m[0][2],
        M.m[1][0], M.m[1][1], M.m[1][2], det,
        M.m[2][0], M.m[2][1], M.m[2][2], fabsf(det),
        v.x, v.y, v.z, v3Len(v),
        Mv.x, Mv.y, Mv.z, v3Len(Mv)];

    _hud.stringValue = text;
}

// ============================================================
#pragma mark - Events
// ============================================================

- (void)mouseDown:(NSEvent *)e    { _lastMouse = e.locationInWindow; }
- (void)mouseUp:(NSEvent *)e      {}
- (void)mouseDragged:(NSEvent *)e {
    NSPoint p = e.locationInWindow;
    _rotY += (p.x - _lastMouse.x) * 0.5f;
    _rotX -= (p.y - _lastMouse.y) * 0.5f;
    if (_rotX > 89) _rotX = 89;
    if (_rotX < -89) _rotX = -89;
    _lastMouse = p;
    [self setNeedsDisplay:YES];
}

- (void)scrollWheel:(NSEvent *)e {
    _zoom -= e.deltaY * 0.3f;
    if (_zoom < 2.0f) _zoom = 2.0f;
    if (_zoom > 20.0f) _zoom = 20.0f;
    [self setNeedsDisplay:YES];
}

- (void)keyDown:(NSEvent *)e {
    unichar c = [e.characters characterAtIndex:0];
    switch (c) {
        case '1': _mode = 1; _t = 0; _tDir = 1; break;
        case '2': _mode = 2; _t = 0; break;
        case '3': _mode = 3; _t = 0; _tDir = 1; break;
        case ' ': _paused = !_paused; break;
        case 'r': case 'R': _t = 0; _tDir = 1; break;
        case 'v': case 'V': _rotX = 25; _rotY = -35; _zoom = 7; break;
        default: break;
    }
    [self setNeedsDisplay:YES];
}

@end

// ============================================================
#pragma mark - App Delegate
// ============================================================

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property (strong) NSWindow *window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)n {
    NSRect frame = NSMakeRect(100, 100, 1100, 750);
    _window = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                  NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
        backing:NSBackingStoreBuffered defer:NO];
    _window.title = @"3D Matrix Transformation Demo";
    _window.minSize = NSMakeSize(640, 480);

    GLView *gl = [[GLView alloc] initWithFrame:_window.contentView.bounds];
    gl.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _window.contentView = gl;

    [_window makeKeyAndOrderFront:nil];
    [_window makeFirstResponder:gl];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)a { return YES; }

@end

// ============================================================
#pragma mark - Main
// ============================================================

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Minimal menu for Cmd+Q
        NSMenu *bar = [[NSMenu alloc] init];
        NSMenuItem *appItem = [[NSMenuItem alloc] init];
        [bar addItem:appItem];
        NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"MatrixDemo"];
        [appMenu addItemWithTitle:@"Quit" action:@selector(terminate:) keyEquivalent:@"q"];
        appItem.submenu = appMenu;
        [app setMainMenu:bar];

        AppDelegate *del = [[AppDelegate alloc] init];
        app.delegate = del;
        [app activateIgnoringOtherApps:YES];
        [app run];
    }
    return 0;
}
