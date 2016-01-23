TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += C:/opencv/build/install/include
LIBS += "C:/opencv/build/bin/*.dll"

SOURCES += main.cpp \
    argparser.cpp \
    backgroundsubsegmenter.cpp \
    pedestrianbuilder.cpp \
    tracker.cpp

HEADERS += \
    argparser.h \
    backgroundsubsegmenter.h \
    pedestrianstructure.h \
    pedestrianbuilder.h \
    tracker.h

