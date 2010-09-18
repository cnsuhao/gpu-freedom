program testTwoDays;

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
  WriteLn('Initial conditions created');

  initPlot;

  printEarthStatus(2000, 0, 0, clima, world);
  printPlanet(2000,0, 0, clima, world, false, false);

  WriteLn('Simulating 3 days of weather including energy cycle, winds,');
  WriteLn('marine currents, steam generation and rain');
  WriteLn('Result will be stored in text file '+FILENAME_EVOLUTION);
  WriteLn('and '+FILENAME_PLANET);

  for day := 170 to 172  do
  begin
    Write(IntToStr(day)+' ');

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
              //printPlanet(2000, day, hour, clima, world, true, false);
              moveSteam(@clima.wind, @clima.steam, @tmpGrid);


            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                    rainSteam(clima, world, i, j);
                    waterOrIce(clima, world, i, j);
                    updateOutgoingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                  end;

         printEarthStatus(2000, day, hour, clima, world);
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
  printPlanet(t.year, t.day, t.hour, clima, world, false, false);
  WriteLn('Simulation finished :-)');

end.
