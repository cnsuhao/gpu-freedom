unit plotBlueBall;


interface

uses
  Classes, SysUtils, datastructure, conversion, averages;

const
    FILENAME_EVOLUTION = 'plugins\output\simclimate_evolution.txt';
    FILENAME_PLANET    = 'plugins\output\simclimate_planet.txt';

procedure initPlot;
procedure printPlanet(year, day, hour : Longint; var clima : TClima; var world : TWorld; plotWind : Boolean; overwrite : Boolean);
procedure printEarthStatus(year, day, hour : Longint; var clima : TClima; var world : TWorld);


implementation


var F, G : Textfile;

procedure initPlot;
begin
  AssignFile(F, FILENAME_EVOLUTION);
  Rewrite(F);
  CloseFile(F);
  AssignFile(G, FILENAME_PLANET);
  Rewrite(G);
  CloseFile(G);
end;

procedure printEarthStatus(year, day, hour : Longint; var clima : TClima; var world : TWorld);

begin
   AssignFile(F, FILENAME_EVOLUTION);
   Append(F);
   WriteLn(F, 'On year '+IntToStr(year)+' day '+IntToStr(day)+' at hour '+IntToStr(hour)+' the temperatures are ');
   WriteLn(F, 'T overall                       :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, AVERAGE))));
   WriteLn(F, 'T atmosphere                    :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, ATMOSPHERE))));
   WriteLn(F, 'T ocean and emerged surface     :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, OCEAN_TERR))));
   WriteLn(F, 'T ocean                         :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, OCEAN))));
   WriteLn(F, 'T emerged surface               :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, TERRAIN))));
   WriteLn(F, 'T air over ocean                :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, AIR_OVER_OCEAN))));
   WriteLn(F, 'T air over emerged surface      :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, AIR_OVER_TERRAIN))));
   WriteLn(F, 'Average humidity                :  '+FloatToStr(computeAvgHumidity(world, clima)));
   WriteLn(F, 'Average steam on air (kg/deg2)  :  '+FloatToStr(computeAvgSteamOnAir(clima)));
   WriteLn(F, 'Average rain (kg/deg2)          :  '+FloatToStr(computeAvgRain(clima)));
   WriteLn(F, 'Average wind movements          :  '+FloatToStr(computeAvgWindMovements(clima)));
   WriteLn(F, 'Average marine current movements:  '+FloatToStr(computeAvgCurrentMovements(world, clima)));
   WriteLn(F, 'Squares covered with ice        :  '+IntToStr(computeIceCoverage(clima, NONE)));
   WriteLn(F, 'Squares covered with ice (North):  '+IntToStr(computeIceCoverage(clima, NORTH)));
   WriteLn(F, 'Squares covered with ice (South):  '+IntToStr(computeIceCoverage(clima, SOUTH)));
   WriteLn(F);
   CloseFile(F);
end;

function plotWindChar(i, j : Longint; var clima : TClima; var world : TWorld) : String;
var
 char : String;
begin
             if clima.wind[0][i][j]=WEST then char := '<'
             else
             if clima.wind[0][i][j]=EAST then char := '>'
             else
             if clima.wind[0][i][j]=NORTH then char := '^'
             else
             if clima.wind[0][i][j]=SOUTH then char := 'v'
             else
             if clima.wind[0][i][j]=NORTH_WEST then char := '\'
             else
             if clima.wind[0][i][j]=NORTH_EAST then char := '/'
             else
             if clima.wind[0][i][j]=SOUTH_WEST then char := '/'
             else
             if clima.wind[0][i][j]=SOUTH_EAST then char := '\';

             Result := char;
end;

function plotSteamAndRainChar(i, j : Longint; var clima : TClima; var world : TWorld) : String;
var
 char : String;
begin
      if world.isOcean[i][j] then
          char := ' '
        else
          char := '#';

    if clima.isIce[i][j] then char := 'I';

    if clima.humidity[i][j]>0.98 then char := 'O'
    else
    if clima.humidity[i][j]>0.95 then char := 'o'
    else
    //if clima.humidity[i][j]>0.4 then char := ':'
    //else
    if clima.humidity[i][j]>0.87 then char := '.';

    if clima.rain[i][j] and (not world.isOcean[i][j]) then char := '/';

   Result := char;
end;

function plotChar(i, j : Longint; var clima : TClima; var world : TWorld; plotWind : Boolean) : String;
begin
  if plotWind then
     Result := plotWindChar(i, j, clima, world)
    else
     Result := plotSteamAndRainChar(i, j, clima, world);
end;

procedure printPlanet(year, day, hour : Longint; var clima : TClima; var world : TWorld; plotWind : Boolean; overwrite : Boolean);
var i, j : Longint;
begin
   AssignFile(G, FILENAME_PLANET);
   if overwrite then Rewrite(G) else Append(G);
   WriteLn(G); WriteLn(G); WriteLn(G);
   WriteLn(G, 'Planet on year '+IntToStr(year)+' day '+IntToStr(day)+' at hour '+IntToStr(hour));
   for j:=0 to 179 do
    begin
     for i:=0 to 359 do
        begin
          Write(G, plotChar(i, j, clima, world, plotWind));
        end;
     WriteLn(G);
    end;

   CloseFile(G);
end;




end.

