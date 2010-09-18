program testSteamMovements;

{$APPTYPE CONSOLE}

uses
  SysUtils, energyfunctions,conversion, datastructure, flux, statechanges,
  watercycle, climaconstants, plotBlueBall, initmodel, riverandlakes;

var clima : TClima;
    world : TWorld;
    t     : TTime;
    s     : TSolarSurface;
    day,
    hour,
    i, j  : Longint;
    earthInclination : TClimateType;

    tmpGrid : TGrid;

begin
 try
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, TInitCond.thermic_poles, TInitCond.thermic_poles);
  WriteLn('Init time and solar surface');
  initTime(t, s);
  WriteLn('Creating a line of steam');
  for j:=0 to 179 do
   begin
    clima.steam[0][60][j] := MaxWaterInSteam(clima, world, 0, 60, j);
    computeHumidity(clima, world, 60, j);
   end;
  WriteLn('Initial conditions created');
  WriteLn('Result will be stored in text file '+FILENAME_EVOLUTION);
  WriteLn('and '+FILENAME_PLANET);

  initPlot;

  printEarthStatus(0, 0, 0, clima, world);
  printPlanet(0, 0, 0, clima, world, false, true);
  moveEnergy(clima, @clima.energy_atmosphere, @tmpGrid, @clima.T_atmosphere, @clima.wind, WIND, world, true);
  moveEnergy(clima, @clima.energy_ocean_terr, @tmpGrid, @clima.T_ocean_terr, @clima.surfaceTransfer, SURFACE_AND_MARINE_CURRENT, world, true);
  printPlanet(2000, 0, 0, clima, world, true, false);


  for day := 170 to 170 do
  begin
    Write(IntToStr(day)+' ');

    for hour := 0 to 23 do
       begin
         Write('.');
         clearRain(clima);

         moveSteam(@clima.wind, @clima.steam, @tmpGrid);
         for j:=0 to 179 do
           for i:=0 to 359 do
             computeHumidity(clima, world, i, j);

         printPlanet(2000, day, hour, clima, world, false, false);

         stepTime(t,s);
       end;

      WriteLn;
   end;

  WriteLn;

 except
    on E : Exception do
      WriteLn('Exception: '+e.Message);
 end;

  WriteLn('Simulation finished :-)');

end.
