unit mainloop;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, energyfunctions,conversion, datastructure, flux, statechanges,
  watercycle, climaconstants, plotBlueBall, riverandlakes,
  averages, initmodel, vulcanandbombs, co2cycle;

procedure mainSimTimestepLoop(var world : TWorld; var clima : TClima; var s : TSolarSurface;
                              var t : TTime; var tmpGrid : TGrid);

procedure mainSimDayLoop(var world : TWorld; var clima : TClima; var s : TSolarSurface;
                         var t : TTime; var tmpGrid : TGrid);


implementation


procedure mainSimTimestepLoop(var world : TWorld; var clima : TClima; var s : TSolarSurface; var t : TTime; var tmpGrid : TGrid);
var l, i, j : Longint;
    earthInclination : TClimateType;
begin
         clearRain(clima);
         earthInclination := computeEarthInclination(t.day);
            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                   updateIncomingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                   formSteam(clima, world, s, i, j, t.day); // add steam to layer 0
                   for l:=1 to TMdlConst.atmospheric_layers-1 do    // move steam from layer l-1 to layer l
                     moveSteamUp(clima, world, s, l, i, j);
                   formCO2(clima, world, i, j);
                  end;

            moveEnergy(clima, @clima.energy_ocean_terr, @tmpGrid, @clima.T_ocean_terr, @clima.surfaceTransfer, SURFACE_AND_MARINE_CURRENT, world, true);
            moveEnergy(clima, @clima.energy_atmosphere, @tmpGrid, @clima.T_atmosphere[0][0][0], @clima.wind[0][0][0], WIND, world, true);

            applyCoriolis(@clima.wind[0], @clima.wind[0], TMdlConst.rotation);
            for l:=1 to TMdlConst.atmospheric_layers-1 do
                  applyCoriolis(@clima.wind[l-1], @clima.wind[l],  TMdlConst.rotation);

            followSurface(world, @clima.wind[0]);

            for l:=0 to TMdlConst.atmospheric_layers-1 do
                  moveSteam(@clima.wind[l], @clima.steam[l], @tmpGrid);

            moveCO2(@clima.wind[0], @clima.co2_tons[0], @tmpGrid);

            vulcanoEruption(world, clima);
            singleNuclearBomb(world, clima);
            nuclearWar(world, clima);
            moveAshes(@clima.wind[0][0][0], @clima.ashes_pct[0][0], @tmpGrid);
            ashesFallDown(world, clima);

            for j := 0 to 179 do
              for i:=0 to 359 do
                  begin
                    for l:=TMdlConst.atmospheric_layers-1 downto 1 do    // move steam from layer l to layer l-1
                        moveSteamDown(clima, world, s, l, i, j);
                    rainSteam(clima, world, i, j);
                    waterOrIce(clima, world, i, j);
                    absorbCO2(clima, world, s, i, j);
                    updateOutgoingEnergyOnCellGrid(clima, world, s, earthInclination, i, j);
                  end;

         moveWaterDownToOcean(world, clima, @clima.water_surface, @tmpGrid);
         stepTime(t,s);
end;


procedure mainSimDayLoop(var world : TWorld; var clima : TClima; var s : TSolarSurface; var t : TTime; var tmpGrid : TGrid);
begin
       increasePopulation(clima, 1);
       printEarthStatus(t.year, t.day, Round(t.hour), clima, world);
       printPlanet(t.year, t.day, Round(t.hour), clima, world, false, true);
       if (t.day mod TSimConst.decrease_rain_times = 0) then decreaseRainTimes(clima);
end;

end.

