CC = arch -arm64 clang
CFLAGS = -fobjc-arc -Wno-deprecated-declarations
FRAMEWORKS = -framework Cocoa -framework OpenGL
TARGET = LinRegDemo

all: $(TARGET)

$(TARGET): linreg.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $(TARGET) linreg.m

clean:
	rm -f $(TARGET)

.PHONY: all clean
