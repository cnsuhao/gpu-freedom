unit averages;

{$mode objfpc}{$H+}

interface

uses
  datastructure, climaconstants, statechanges, Classes, SysUtils;

  function Avg(one, two : TClimateType) : TClimateType;

  function computeAvgKTemperature(var w : TWorld; var clima : TClima; tType : Longint) : TClimateType;
  function computeAvgSteamOnAir(var clima : TClima) : TClimateType;
  function computeAvgRain(var clima : TClima) : TClimateType;
  function computeAvgHumidity(var w : TWorld; var clima : TClima): TClimateType;
  function computeAvgWindMovements(var clima : TClima) : TClimateType;
  function computeAvgCurrentMovements(var w : TWorld; var clima : TClima) : TClimateType;
  function computeIceCoverage(var clima : TClima; direction : ShortInt): Longint;
  procedure computeAvgWaterSurface(var w : TWorld; var clima : TClima);

implementation

function Avg(one, two : TClimateType) : TClimateType;
begin
  Result := (one+two)/2;
end;


function computeAvgKTemperature(var w : TWorld; var clima : TClima; tType : Longint) : TClimateType;
var i, j,
    count : Longint;
    temp : TClimateType;
begin
 temp := 0;
 count := 0;
 for j:=0 to 179 do
   for i:=0 to 359 do
     begin
       if tType = ATMOSPHERE then
         temp := clima.T_atmosphere[0][i][j]+ temp
       else
       if tType = OCEAN_TERR then
         temp := clima.T_ocean_terr[i][j]+ temp
       else
       if tType = AVERAGE then
         temp := clima.T_atmosphere[0][i][j]+ clima.T_ocean_terr[i][j] + temp
       else
       if ((tType = OCEAN) and w.isOcean[i][j]) then
        begin
         temp := clima.T_ocean_terr[i][j]+ temp;
         Inc(count);
        end
       else
       if ((tType = TERRAIN) and (not w.isOcean[i][j])) then
        begin
         temp := clima.T_ocean_terr[i][j]+ temp;
         Inc(count);
        end
       else
       if ((tType = AIR_OVER_OCEAN) and w.isOcean[i][j]) then
        begin
         temp := clima.T_atmosphere[0][i][j]+ temp;
         Inc(count);
        end
       else
       if ((tType = AIR_OVER_TERRAIN) and (not w.isOcean[i][j])) then
        begin
         temp := clima.T_atmosphere[0][i][j]+ temp;
         Inc(count);
        end;
     end;

 if tType = AVERAGE then
   Result := temp / (180*360*2)
 else
 if count>0 then
   Result := temp / count
 else
   Result := temp / (180*360);
end;

function computeAvgSteamOnAir(var clima : TClima) : TClimateType;
var l, i, j : Longint;
    steam : TClimateType;
begin
 steam := 0;
 for l:=0 to TMdlConst.atmospheric_layers-1 do
  for j:=0 to 179 do
   for i:=0 to 359 do
      steam := steam + clima.steam[l][i][j];
 Result := steam / (360 * 180 * TMdlConst.atmospheric_layers);
end;

function computeAvgRain(var clima : TClima): TClimateType;
var i, j : Longint;
    rainC : Longint;
begin
 rainC := 0;
 for j:=0 to 179 do
  for i:=0 to 359 do
      if clima.rain[i][j] then Inc(rainC);
 Result := rainC / (360 * 180);
end;

function computeAvgHumidity(var w : TWorld; var clima : TClima): TClimateType;
var i, j : Longint;
    humidity : TClimateType;
begin
 humidity := 0;
 for j:=0 to 179 do
  for i:=0 to 359 do
      humidity := humidity + computeHumidity(clima, w, i, j);

 Result := humidity / (360 * 180);
end;

function computeAvgWindMovements(var clima : TClima) : TClimateType;
var i, j : Longint;
    movements : Longint;
begin
 movements := 0;
 for j:=0 to 179 do
  for i:=0 to 359 do
   if (clima.wind[0][i][j]<>NONE) then
     movements := movements + 1;

 Result := movements/(360*180);
end;

function computeAvgCurrentMovements(var w : TWorld; var clima : TClima) : TClimateType;
var i, j : Longint;
    movements, count : Longint;

begin
 movements := 0;
 count := 0;
 for j:=0 to 179 do
  for i:=0 to 359 do
   if (clima.surfaceTransfer[i][j]<>NONE) and (w.isOcean[i][j]) then
    begin
     movements := movements + 1;
     count := count + 1;
    end;
 if (count<>0) then
  Result := movements/count
 else
  Result := 0;
end;

function computeIceCoverage(var clima : TClima; direction : ShortInt): Longint;
var i, j : Longint;
    iceCoverage : Longint;
begin
 iceCoverage := 0;
 for j:=0 to 179 do
  for i:=0 to 359 do
     if clima.IsIce[i][j] then
      begin
        if (direction = NORTH) and (j<90) then
          iceCoverage := iceCoverage + 1
        else
        if (direction = SOUTH) and (j>=90) then
          iceCoverage := iceCoverage + 1
        else
        if (direction = NONE) then
          iceCoverage := iceCoverage + 1;
      end;

 Result := iceCoverage;
end;


procedure computeAvgWaterSurface(var w : TWorld; var clima : TClima);
var i, j, count : Longint;
    avg : TClimateType;
begin
 for j:=0 to 179 do
  begin
     avg := 0;
     count := 0;

     for i:=0 to 359 do
      begin
       if (not w.isOcean[i][j]) and (clima.water_surface[i][j]>0) then
        begin
          avg := avg + clima.water_surface[i][j];
          Inc(count);
        end;
      end;

       if count<>0 then clima.avgWaterSurface[j] := avg/count
       else clima.avgWaterSurface[j] := 0;
  end;
end;


end.

