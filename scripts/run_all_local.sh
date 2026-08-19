#!/usr/bin/env bash
# Chay TOAN BO chuong trinh do lai tren MOT may (Apple M5, 10 nhan, 16 GB).
#
# VI SAO MOT MAY: sổ ghi luat "moi so cua mot bai phai tu MOT may" sau ca 17/08
# (cung script cung seed cho 0,1198 tren Linux va 0,1140 tren Mac). Ban da nop
# chay tren RTX 4090; neu chi chay bo sung o day thi so moi va so cu khong so
# duoc voi nhau. Nen chay lai tat ca o day.
#
# VI SAO MAY NAY CHU KHONG PHAI GPU: do duoc 57 giay/run o day so voi 167 giay
# tren RTX 4090. Mo hinh chi 3,3k tham so nen phi khoi dong kernel GPU lan at
# phan tinh toan. Day cung la du kien tra loi R1#5 va R2#4 ve phan cung.
#
# OMP_NUM_THREADS=1 la BAT BUOC: mac dinh moi tien trinh numpy mo toi 10 luong,
# sau tien trinh thanh 60 luong tranh 10 nhan va cham hon chay tuan tu.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
W=${W:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$1"; }

log "1/4  sweep lr, BoT-IoT, binary, Dir 0.1  (240 run)"
$PY scripts/lr_sweep.py --workers $W --config e1_botiot --mode binary \
    --output-root results/lrsweep_botiot --skip-existing --python $PY

log "2/4  sweep lr, ToN-IoT, binary, Dir 0.1  (240 run)"
$PY scripts/lr_sweep.py --workers $W --config e1_toniot --mode binary \
    --output-root results/lrsweep_toniot --skip-existing --python $PY

log "3/4  sweep lr, CSE-CIC, binary, Dir 0.1  (240 run)"
$PY scripts/lr_sweep.py --workers $W --config e1_cseciic --mode binary \
    --output-root results/lrsweep_cseciic --skip-existing --python $PY

# O nay sinh ra tuyen bo +1,73 pp p=0,012, ket qua duong manh nhat cua bai.
log "4/4  sweep lr, CSE-CIC, MULTICLASS, Dir 0.1  (240 run)"
$PY scripts/lr_sweep.py --workers $W --config e1_cseciic --mode multiclass \
    --output-root results/lrsweep_cseciic_mc --skip-existing --python $PY

log "XONG TAT CA"
for d in botiot toniot cseciic cseciic_mc; do
  n=$(find results/lrsweep_$d -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  results/lrsweep_$d : $n/240 run"
done
