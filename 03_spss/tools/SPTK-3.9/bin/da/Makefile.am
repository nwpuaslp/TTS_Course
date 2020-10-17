if USEDA
AM_CPPFLAGS = -I @top_srcdir@/include
bin_PROGRAMS = dawrite
dawrite_SOURCES = da.h dawrite.c winplay.c winplay.h
dawrite_LDADD = @top_builddir@/lib/libSPTK.a

bin_SCRIPTS = da
EXTRA_DIST = da.in

CLEANFILES = $(bin_PROGRAMS) $(bin_SCRIPTS)

replace_path = sed \
               -e 's,[@]bindir[@],$(bindir),g' \
               -e 's,[@]pkgdatadir[@],$(pkgdatadir),g' \
               -e 's,[@]package_version[@],$(PACKAGE_VERSION),g' \
               -e 's,[@]csh[@],$(CSH),' \
               -e 's,[@]io_symbol[@],$(IO_SYMBOL),g' \
               -e 's,[@]io_format[@],$(IO_FORMAT),g'

da: da.in
	$(replace_path) < da.in > $@
	chmod +x $@
endif
