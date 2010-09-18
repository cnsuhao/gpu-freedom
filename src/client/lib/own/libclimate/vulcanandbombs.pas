unit vulcanandbombs;

{$mode objfpc}{$H+}

interface

uses
  climaconstants, datastructure, conversion, Classes, SysUtils, watercycle,
  energyfunctions, flux;

const
  MEGATON_TO_JOULE = 4.184E15;

procedure vulcanoEruption(var w : TWorld; var clima : TClima);
procedure singleNuclearBomb(var w : TWorld; var clima : TClima);
procedure nuclearWar(var w : TWorld; var clima : TClima);
procedure moveAshes(wind : PGridShortInt; ashes : PGrid; copyAshes : PGrid);
procedure ashesFallDown(var w : TWorld; var clima : TClima);


implementation

procedure vulcanoEruption(var w : TWorld; var clima : TClima);
var i, j : Longint;
begin
  if not TSpecialParam.vulcan then Exit;
  TSpecialParam.ashes_cycle_active := true;

  i := LonToX(TSpecialParam.vulcan_lon);
  j := LatToY(TSpecialParam.vulcan_lat);

  clima.ashes_pct[i][j] := clima.ashes_pct[i][j]+TSpecialParam.vulcan_ashes_pct*(TSimConst.degree_step/15);
  clima.co2_tons[i][j]  := clima.co2_tons[i][j]+TSimConst.co2_production_vulcano*(TSimConst.degree_step/15)/(365*24);
  TSpecialParam.vulcan_hours:=TSpecialParam.vulcan_hours-TSimConst.degree_step/15;
  TSpecialParam.vulcan := (TSpecialParam.vulcan_hours>0);
  if not TSpecialParam.vulcan then TSpecialParam.vulcan_hours:=0; // to avoid rounding errors
end;

procedure launchNuclearBomb(var w : TWorld; var clima : TClima; i, j : Longint);
var energy : TClimateType;
begin
  // bomb launch
  clima.ashes_pct[i][j] := clima.ashes_pct[i][j]+TSpecialParam.nuclear_ashes_pct*(TSimConst.degree_step/15);
  clima.population[i][j] := 0;

  energy := TSpecialParam.nuclear_bomb_energy*MEGATON_TO_JOULE;
  clima.energy_atmosphere[i][j] := clima.energy_atmosphere[i][j]+energy;
  updateTemperature(clima, w, i, j);
end;

procedure singleNuclearBomb(var w : TWorld; var clima : TClima);
var i, j : Longint;
begin
  if not TSpecialParam.nuclear_bomb then Exit;
  TSpecialParam.ashes_cycle_active := true;

  i := LonToX(TSpecialParam.nuclear_bomb_lon);
  j := LatToY(TSpecialParam.nuclear_bomb_lat);
  launchNuclearBomb(w, clima, i, j);

  TSpecialParam.nuclear_bomb := false; // single launch
end;

procedure nuclearWar(var w : TWorld; var clima : TClima);
var i, j,
    l : Longint;

    lat : TClimateType;
begin
  if not TSpecialParam.nuclear_war then Exit;
  TSpecialParam.ashes_cycle_active := true;

  for l:=1 to 10 do
     begin
      i := Trunc(Random * 359.0);
      j := Trunc(Random * 179.0);

      if (w.isOcean[i][j]) then continue;
      lat := YtoLat(j);
      if Abs(lat)<30 then continue;

      launchNuclearBomb(w, clima, i, j);
     end;

  TSpecialParam.nuclear_war_hours:=TSpecialParam.nuclear_war_hours-TSimConst.degree_step/15;
  TSpecialParam.nuclear_war := (TSpecialParam.nuclear_war_hours>0);
  // to avoid rounding errors
  if not TSpecialParam.nuclear_war then TSpecialParam.nuclear_war_hours:=0;
end;

procedure moveAshes(wind : PGridShortInt; ashes : PGrid; copyAshes : PGrid);
begin
  if not TSpecialParam.ashes_cycle_active then Exit;
  moveParticlesInAtm(wind, ashes, copyAshes);
end;

procedure ashesFallDown(var w : TWorld; var clima : TClima);
var activity : Boolean;
    i, j : Longint;
begin
  if not TSpecialParam.ashes_cycle_active then Exit;
  for j := 0 to 179  do
      for i:=0 to 359 do
       begin
         // we do not allow percentages over 1
         if (clima.ashes_pct[i][j]>1) then clima.ashes_pct[i][j]:=1;
         clima.ashes_pct[i][j] := clima.ashes_pct[i][j] * (1-TSpecialParam.ashes_fallout_pct*(TSimConst.degree_step/15));
       end;
end;


end.

