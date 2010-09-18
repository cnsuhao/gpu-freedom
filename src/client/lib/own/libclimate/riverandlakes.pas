unit riverandlakes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, datastructure, climaconstants, conversion, Math, averages;

procedure clearRain(var clima : TClima);
procedure moveWaterDownToOcean(var w : TWorld; var clima : TClima; water_surface : PGrid; water_grid : PGrid);
function isRiver(var clima : TClima; i, j : Longint) : Boolean;
function isGrass(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
function isForest(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
function isJungle(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
function isDesert(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;


implementation

procedure clearRain(var clima : TClima);
var i, j : Longint;
begin
  for j := 0 to 179 do
    for i := 0 to 359 do
      clima.rain[i][j] := false;
end;


procedure moveWaterDownToOcean(var w : TWorld; var clima : TClima; water_surface : PGrid; water_grid : PGrid);
var
  i,
  j,
  check_north,
  check_south,
  check_east,
  check_west : Longint;

  E_own,
  E_north,
  E_south,
  E_west,
  E_east,
  E_north_west,
  E_north_east,
  E_south_west,
  E_south_east : TClimateType;

  E_lowestCardinal,
  E_lowestDiagonal,
  E_lowest : TClimateType;

  procedure transferWater(i, j, i_target, j_target : Longint);
  var transferredWater : TClimateType;
  begin
   transferredWater :=  water_grid^[i][j] * TSimConst.riverandlakes_pct;
   water_surface^[i][j] := water_surface^[i][j] - transferredWater;
   if (water_surface^[i][j]<0) then water_surface^[i][j] := 0;

   if clima.IsIce[i][j] or w.isOcean[i][j] then Exit; // water reached its target


   // water continues to flow down...
   water_surface^[i_target][j_target] := water_surface^[i_target][j_target] + transferredWater;
   if water_surface^[i_target][j_target]>1E100 then //raise Exception.Create('Too much water!');
             water_surface^[i_target][j_target] := 1E100; // we limit the quantity of water in this way
  end;

begin
  // we need a local copy of the energy grid
  for j:=0 to 179 do
   for i:=0 to 359 do
      water_grid^[i][j] := water_surface^[i][j];

 for j:=0 to 179 do
  for i:=0 to 359 do
      begin
        if w.isOcean[i][j] or clima.isIce[i][j] then continue;

        check_north := j-1;
        check_south := j+1;
        check_west  := i-1;
        check_east  := i+1;

        // we live on a sphere
        if check_north<0 then check_north := 179;
        if check_south>179 then check_south := 0;
        if check_west<0 then check_west := 359;
        if check_east>359 then check_east := 0;

        E_north := w.elevation[i][check_north];
        E_south := w.elevation[i][check_south];
        E_west  := w.elevation[check_west][j];
        E_east  := w.elevation[check_east][j];
        E_north_west := w.elevation[check_west][check_north];
        E_north_east := w.elevation[check_east][check_north];
        E_south_west := w.elevation[check_west][check_south];
        E_south_east := w.elevation[check_east][check_south];
        E_own   :=  w.elevation[i][j];

        E_lowestCardinal  := Math.Min(Math.min(E_north, E_south), Math.Min(E_west, E_east));
        E_lowestDiagonal := Math.Min(Math.min(E_north_east, E_south_west), Math.Min(E_north_west, E_south_east));
        E_lowest := Math.Min(E_lowestCardinal, E_lowestDiagonal);

        if (E_own = E_lowest) then
           transferWater(i , j, i, j)
        else
        if (E_west = E_lowest) then
              transferWater(i , j, check_west, j)
         else
         if (E_east = E_lowest) then
              transferWater(i , j, check_east, j)
         else
         if (E_north = E_lowest) then
              transferWater(i , j, i, check_north)
         else
         if (E_south = E_lowest) then
              transferWater(i , j, i, check_south)
         else
         if (E_north_west = E_lowest) then
              transferWater(i , j, check_west, check_north)
         else
         if (E_north_east = E_lowest) then
              transferWater(i , j, check_east, check_north)
         else
         if (E_south_east = E_lowest) then
              transferWater(i , j, check_east, check_south)
         else
         if (E_south_west = E_lowest) then
              transferWater(i , j, check_west, check_south);
     end;

  computeAvgWaterSurface(w, clima);
end;

function isRiver(var clima : TClima; i, j : Longint) : Boolean;
begin
  Result := clima.water_surface[i][j]>TSimConst.paint_river_pct * clima.avgWaterSurface[j];
end;

function isGrass(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
begin
  Result := (clima.water_surface[i][j]>=1);
end;

function isForest(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
var T : TClimateType;
begin
  T := KtoC(clima.T_ocean_terr[i][j]);
  Result := (clima.water_surface[i][j]<=TSimConst.paint_river_pct * clima.avgWaterSurface[j]) and (T>0) and (T<25) and
            (clima.rain_times[i][j]>1);
end;

function isJungle(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
var T : TClimateType;
begin
  T := KtoC(clima.T_ocean_terr[i][j]);
  Result := (clima.water_surface[i][j]<=TSimConst.paint_river_pct * clima.avgWaterSurface[j])
            and (T>=25) and (T<50) and (clima.rain_times[i][j]>2);
end;

function isDesert(var w : TWorld; var clima : TClima; i, j : Longint) : Boolean;
var T : TClimateType;
begin
  T := KtoC(clima.T_ocean_terr[i][j]);
  Result := (T>=50) and (clima.water_surface[i][j]<1);
end;


end.

