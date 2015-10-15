# ../w32/glpsol.exe -m pp3.mod -d pp3.dat
# like pp2 but a lake more and a flow plant more
param Ptur_max;
param Ptur2_max;

param Ppum_max;
param Qw_tur;
param Qw_tur2;
param Qw_pum;

param lake_max;
param lake2_max;

set T:= 1..24;
set Tplus1 := 1..25;

param hpfc{T};

display Ptur_max;
display Ptur2_max;
display Ppum_max;
display Qw_tur;
display Qw_pum;
display lake_max;
display lake2_max;
display hpfc;


var sTur{T},  >=0;
var sTur2{T}, >=0;

var sPum{T}, >=0;
var lLevel{Tplus1}, >=0;
var lLevel2{Tplus1}, >=0;

var turActive{T}, binary;
var pumActive{T}, binary;

s.t. level_start_a : lLevel[1]=lake_max;
s.t. level_start_b : lLevel2[1]=lake2_max/2;

s.t. level_a {t in T}: lLevel[t]<=lake_max;
s.t. level_b {t in T}: lLevel2[t]<=lake2_max;

s.t. avail_a {t in T}: sTur[t]<=Ptur_max;
s.t. avail_b {t in T}: sPum[t]<=Ppum_max;
s.t. avail_c {t in T}: sTur2[t]<=Ptur2_max;

s.t. chain_a_1 {t in T}: lLevel[t+1] = lLevel[t]  - sTur[t]/Ptur_max*Qw_tur*3600+ sPum[t]/Ppum_max*Qw_pum*3600;
s.t. chain_a_2 {t in T}: turActive[t]+pumActive[t]<=1;
s.t. chain_a_3 {t in T}: sTur[t] <= turActive[t]*Ptur_max;
s.t. chain_a_4 {t in T}: sPum[t] <= pumActive[t]*Ppum_max;

s.t. chain_b_1 {t in T}: lLevel2[t+1] = lLevel2[t] + sTur[t]/Ptur_max*Qw_tur - sPum[t]/Ppum_max*Qw_pum*3600 - sTur2[t]/Ptur2_max*Qw_tur2;

maximize profit: sum{t in T} (sTur[t]*hpfc[t]-sPum[t]*hpfc[t]+sTur2[t]*hpfc[t]);

solve;
display lLevel;
display sTur;
display sPum;
display lLevel2;
display sTur2;
end;