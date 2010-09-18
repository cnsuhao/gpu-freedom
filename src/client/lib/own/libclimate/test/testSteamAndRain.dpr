program testSteamAndRain;

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
    i, j, year  : Longint;
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
  WriteLn('Initial conditions created');

  initPlot;

  printEarthStatus(2000, 0, 0, clima, world);
  printPlanet(2000, 0, 0, clima, world, false, true);

  WriteLn('Simulating 100 days of weather including energy cycle, winds,');
  WriteLn('marine currents, steam generation and rain');
  WriteLn('Result will be stored in text file '+FILENAME_EVOLUTION);
  WriteLn('and '+FILENAME_PLANET);

  for year:=2000 to 2000 do
  for day := 80 to 81  do
  begin
    Write(IntToStr(year)+' '+IntToStr(day)+' ');

    for hour := 0 to 23 do
       begin
         Write('.');
         clearRain(clima);
         earthInclination := computeEarthInclination(day);
            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                   updateIncomingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                   formSteam(clima, world, s, i, j, t.day);
                  end;

              moveEnergy(clima, @clima.energy_atmosphere, @tmpGrid, @clima.T_atmosphere, @clima.wind, WIND, world, true);
              moveEnergy(clima, @clima.energy_ocean_terr, @tmpGrid, @clima.T_ocean_terr, @clima.surfaceTransfer, SURFACE_AND_MARINE_CURRENT, world, true);
              //printPlanet(day, hour, clima, world, true);
              moveSteam(@clima.wind, @clima.steam, @tmpGrid);


            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                    updateTemperature(clima, world, i, j);
                    rainSteam(clima, world, i, j);
                    waterOrIce(clima, world, i, j);
                    updateOutgoingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                  end;

         stepTime(t,s);

       end;

       printEarthStatus(year, day, hour, clima, world);
       printPlanet(year, day, hour, clima, world, false, true);
      WriteLn;
   end;

  WriteLn;

 except
    on E : Exception do
      WriteLn('Exception: '+e.Message);
 end;

  WriteLn('Simulation finished :-)');

end.
