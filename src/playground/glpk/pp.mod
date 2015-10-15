# ../w32/glpsol.exe -m pp.mod -d pp.dat
param Ptur_max;
param Ppum_max;
param Qw_tur;
param Qw_pum;
param lake_max;
set T:= 1..24;
set Tplus1 := 1..25;

param hpfc{T};

display Ptur_max;
display Ppum_max;
display Qw_tur;
display Qw_pum;
display lake_max;
display hpfc;


var sTur{T}, >=0;
var sPum{T}, >=0;
var lLevel{Tplus1}, >=0;

s.t. a {t in T}: lLevel[t]<=lake_max;
s.t. b {t in T}: sTur[t]<=Ptur_max;
s.t. c {t in T}: sPum[t]<=Ppum_max;
s.t. d {t in T}: lLevel[t+1] = lLevel[t]  - sTur[t]/Ptur_max*Qw_tur*3600 + sPum[t]/Ppum_max*Qw_pum*3600;

maximize profit: sum{t in T} (sTur[t]*hpfc[t]-sPum[t]*hpfc[t]);

solve;
display sTur;
display sPum;
display lLevel;
end;