AM_CPPFLAGS = -I @top_srcdir@/include -DLIB=\"$(pkgdatadir)\"
bin_PROGRAMS = us
us_SOURCES = us.c 
us_LDADD = @top_builddir@/lib/libSPTK.a

bin_SCRIPTS = us16 uscd
EXTRA_DIST = us16.in uscd.in

CLEANFILES = $(bin_PROGRAMS) $(bin_SCRIPTS)

dist_pkgdata_DATA = lpfcoef.2to3f \
                    lpfcoef.2to3s \
                    lpfcoef.3to4 \
                    titech/lpfcoef.3to5 \
                    lpfcoef.4to5 \
                    lpfcoef.5to7 \
                    lpfcoef.5to8 \
                    titech/lpfcoef.7to8

replace_path = sed \
               -e 's,[@]bindir[@],$(bindir),g' \
               -e 's,[@]pkgdatadir[@],$(pkgdatadir),g' \
               -e 's,[@]package_version[@],$(PACKAGE_VERSION),g' \
               -e 's,[@]csh[@],$(CSH),' \
               -e 's,[@]io_symbol[@],$(IO_SYMBOL),g' \
               -e 's,[@]io_format[@],$(IO_FORMAT),g'

us16: us16.in
	$(replace_path) < us16.in > $@
	chmod +x $@

uscd: uscd.in
	$(replace_path) < uscd.in > $@
	chmod +x $@
