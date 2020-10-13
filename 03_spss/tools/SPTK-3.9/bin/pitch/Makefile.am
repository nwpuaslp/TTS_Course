AM_CPPFLAGS = -I @top_srcdir@/include
bin_PROGRAMS = pitch
pitch_SOURCES = pitch.c snack/jkGetF0.c snack/sigproc.c snack/jkGetF0.h \
	swipe/swipe.c swipe/vector.c swipe/vector.h
pitch_LDADD = @top_builddir@/lib/libSPTK.a
