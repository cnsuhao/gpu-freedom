program testWindAndCurrents;

{$APPTYPE CONSOLE}

uses
  SysUtils, energyfunctions,conversion, datastructure, flux, riverandlakes, initmodel,
  climaconstants, averages, plotblueball;

var clima : TClima;
    world : TWorld;
    t     : TTime;
    s     : TSolarSurface;
    day,
    hour,
    i, j  : Longint;
    earthInclination : TClimateType;

    winds,
    marine_currents : TGridShortInt;
    tmpGrid : TGrid;

begin
 try
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, TInitCond.thermic_poles, TInitCond.thermic_poles);
  WriteLn('Init time and solar surface');
  initTime(t, s);
  WriteLn('Average temperature is '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, AVERAGE))));
  WriteLn('Initial conditions created');

  WriteLn('Simulating 4 days of weather including energy cycle, winds and marine currents...');
  WriteLn('Result will be stored in text file '+FILENAME_EVOLUTION);
  WriteLn('and '+FILENAME_PLANET);

  for day := 1 to 4 do
  begin
    for hour := 0 to 23 do
       begin
         earthInclination := computeEarthInclination(day);
            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                   updateIncomingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                  end;
         //WriteLn('Generating winds');
         moveEnergy(clima, @clima.energy_atmosphere, @clima.T_atmosphere, @tmpGrid, @winds, WIND, world, true);
         //WriteLn('Generating marine currents');
         moveEnergy(clima, @clima.energy_ocean_terr, @clima.T_ocean_terr, @tmpGrid, @marine_currents, SURFACE_AND_MARINE_CURRENT, world, true);
         stepTime(t,s);
         updateOutgoingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
       end;
   WriteLn('On day '+IntToStr(day)+' the temperatures are ');
   WriteLn('T overall                   :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, AVERAGE))));
   WriteLn('T atmosphere                :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, ATMOSPHERE))));
   WriteLn('T ocean and emerged surface :  '+FloatToStr(KtoC(computeAvgKTemperature(world, clima, OCEAN_TERR))));
   WriteLn;
   ReadLn;
   end;
   
  WriteLn;

 except
    on E : Exception do
      WriteLn('Exception: '+e.Message);
 end;

  ReadLn;

end.
