bin_SCRIPTS = bell gseries glogsp grlogsp gwave mgc2mgclsp mgclsp2mgc raw2wav wav2raw
EXTRA_DIST = bell.in gseries.in glogsp.in grlogsp.in gwave.in mgc2mgclsp.in mgclsp2mgc.in raw2wav.in wav2raw.in

CLEANFILES = $(bin_SCRIPTS)

replace_path = sed \
               -e 's,[@]bindir[@],$(bindir),g' \
               -e 's,[@]package_version[@],$(PACKAGE_VERSION),g' \
               -e 's,[@]csh[@],$(CSH),' \
               -e 's,[@]io_symbol[@],$(IO_SYMBOL),g' \
               -e 's,[@]io_format[@],$(IO_FORMAT),g'

bell:
	$(replace_path) < bell.in > $@
	chmod +x $@

gseries:
	$(replace_path) < gseries.in > $@
	chmod +x $@

glogsp:
	$(replace_path) < glogsp.in > $@
	chmod +x $@

grlogsp:
	$(replace_path) < grlogsp.in > $@
	chmod +x $@

gwave:
	$(replace_path) < gwave.in > $@
	chmod +x $@

mgc2mgclsp:
	$(replace_path) < mgc2mgclsp.in > $@
	chmod +x $@

mgclsp2mgc:
	$(replace_path) < mgclsp2mgc.in > $@
	chmod +x $@

raw2wav:
	$(replace_path) < raw2wav.in > $@
	chmod +x $@

wav2raw:
	$(replace_path) < wav2raw.in > $@
	chmod +x $@
