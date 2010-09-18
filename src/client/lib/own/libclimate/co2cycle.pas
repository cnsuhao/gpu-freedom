unit co2cycle;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, datastructure, climaconstants, watercycle, statechanges, flux, Math;

procedure formCO2(var clima : TClima; var w : TWorld; i, j : Longint);
procedure moveCO2(wind : PGridShortInt; co2 : PGrid; copyCo2 : PGrid);
procedure absorbCO2(var clima : TClima; var w : TWorld; var s : TSolarSurface; i, j : Longint);
procedure increasePopulation(var clima : TClima; days : Longint);

implementation

procedure formCO2(var clima : TClima; var w : TWorld; i, j : Longint);
begin
 if not TSimConst.population then Exit;
 if not TSimConst.energy_source_oil then Exit;
 clima.co2_tons[i][j] := clima.co2_tons[i][j] +
                         TSimConst.co2_production_per_human_per_year/(365/24/15*TSimConst.degree_step)*
                         clima.population[i][j];
end;

procedure moveCO2(wind : PGridShortInt; co2 : PGrid; copyCo2 : PGrid);
begin
  moveParticlesInAtm(wind, co2, copyCO2);
end;

procedure absorbCO2(var clima : TClima; var w : TWorld; var s : TSolarSurface; i, j : Longint);
var absorption ,
    rain_times : TClimateType;
begin
 if  clima.IsIce[i][j] or
    (clima.co2_tons[i][j] = 0) or
    (not isInSunlight(i, j, s))  then Exit;

 if w.isOcean[i][j] then
    // absorption scaled over 12 h per day and over 3 quarters of earth surface
    absorption:=TSimConst.co2_absorption_ocean/(365/12/15*TSimConst.degree_step)/(360*180*3/4)
  else
    begin
      // absorption scaled over 12 h per day and over 1 quarter of earth surface
     absorption:=TSimConst.co2_absorption_vegetation/(365/12/15*TSimConst.degree_step)/(360*180*1/4);
     // jungle absorbs more than desert
     rain_times := clima.rain_times[i][j];
     if rain_times>5 then rain_times := 5;

     absorption := absorption*rain_times;
    end;

  clima.co2_tons[i][j] := clima.co2_tons[i][j] - absorption;
  if (clima.co2_tons[i][j]<0) then clima.co2_tons[i][j] := 0;
end;

procedure increasePopulation(var clima : TClima; days : Longint);
var base, exp, factor : TClimateType;
    i,j    : Longint;
begin
  base := 1;
  exp  := 1 + (TSimConst.population_increase_pct*days/365);
  factor := Power(base, exp);
  for j:=0 to 179 do
    for i:=0 to 359 do
       clima.population[i][j] := clima.population[i][j]*factor;
end;

end.
