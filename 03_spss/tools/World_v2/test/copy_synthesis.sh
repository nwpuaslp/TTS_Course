if [ $# != 1 ]; then
  echo "Usage: copy_synthesis.sh <file>"
  exit 1
fi

file_id=$1

# initializations
fs=16000

if [ "$fs" -eq 16000 ]
then
nFFTHalf=1024 
alpha=0.58
fi

if [ "$fs" -eq 48000 ]
then
nFFTHalf=2048
alpha=0.77
fi

mcsize=59
order=4

### WORLD ANALYSIS -- extract vocoder parameters ###

### extract f0, sp, ap ###
../build/analysis $file_id.wav $file_id.f0 $file_id.sp $file_id.ap

### convert f0 to lf0 ###
x2x +da $file_id.f0 > $file_id.f0a
x2x +af $file_id.f0a | sopr -magic 0.0 -LN -MAGIC -1.0E+10 > $file_id.lf0

### convert sp to mgc ###
x2x +df $file_id.sp | sopr -R -m 32768.0 | mcep -a $alpha -m $mcsize -l $nFFTHalf -e 1.0E-8 -j 0 -f 0.0 -q 3 > $file_id.mgc

### convert ap to bap ###
x2x +df $file_id.ap | sopr -R -m 32768.0 | mcep -a $alpha -m $order -l $nFFTHalf -e 1.0E-8 -j 0 -f 0.0 -q 3 > $file_id.bap

### WORLD Re-synthesis -- reconstruction of parameters ###

### convert lf0 to f0 ###
sopr -magic -1.0E+10 -EXP -MAGIC 0.0 $file_id.lf0 | x2x +fa > $file_id.resyn.f0a
x2x +ad $file_id.resyn.f0a > $file_id.resyn.f0

### convert mgc to sp ###
mgc2sp -a $alpha -g 0 -m $mcsize -l $nFFTHalf -o 2 $file_id.mgc | sopr -d 32768.0 -P | x2x +fd > $file_id.resyn.sp

### convert bap to ap ###
mgc2sp -a $alpha -g 0 -m $order -l $nFFTHalf -o 2 $file_id.bap | sopr -d 32768.0 -P | x2x +fd > $file_id.resyn.ap

../build/synthesis $file_id.resyn.f0 $file_id.resyn.sp $file_id.resyn.ap $file_id.resyn.wav
