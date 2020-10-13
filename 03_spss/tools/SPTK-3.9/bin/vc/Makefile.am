AM_CPPFLAGS = -I hts_engine_API \
              -I @top_srcdir@/include
bin_PROGRAMS = vc
vc_SOURCES = vc.c _vc.c \
            hts_engine_API/HTS_misc.c \
            hts_engine_API/HTS_pstream.c \
            hts_engine_API/HTS_sstream.c \
            hts_engine_API/HTS_hidden.h \
            hts_engine_API/HTS_engine.h
vc_LDADD = @top_builddir@/lib/libSPTK.a
