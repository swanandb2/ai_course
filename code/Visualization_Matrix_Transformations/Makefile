CC = arch -arm64 clang
CFLAGS = -fobjc-arc -Wno-deprecated-declarations
FRAMEWORKS = -framework Cocoa -framework OpenGL
TARGET = MatrixDemo

all: $(TARGET)

$(TARGET): main.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $(TARGET) main.m

clean:
	rm -f $(TARGET)

.PHONY: all clean
