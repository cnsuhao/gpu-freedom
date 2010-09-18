program testEnergyFlow;

{$APPTYPE CONSOLE}

uses
  SysUtils, energyfunctions,conversion, datastructure, initmodel;

var clima : TClima;
    world : TWorld;
    countDay,
    countHour : Longint;


procedure plotTemperature(var clima : TClima; var w : TWorld; i, j : Longint; output : Boolean);
begin
  updateTemperature(clima, w, i, j);
  if output then
    begin
     WriteLn('T atmosphere    : '+FloatToStr(KtoC(clima.T_atmosphere[0][i][j])));
     WriteLn('T ocean/terrain : '+FloatToStr(KtoC(clima.T_ocean_terr[i][j])));
     WriteLn;
    end;
end;

procedure testEnergyFlowOnSquare(lat, lon : TClimateType; day : Longint; var clima : TClima; var w : TWorld; output : Boolean);
var earthInclination,
    energyIn : TClimateType;
    i, j : Longint;
begin
  i := LonToX(lon);
  j := LatToY(lat);

  if output then
    begin
      WriteLn('Energy flow on a degree squared at latitude '+FloatToStr(lat)+' on day '+IntToStr(day));
      Write('Initial conditions on ');
      if (clima.isIce[i][j]) then WriteLn('ICE square:')
      else
      if (w.isOcean[i][j]) then WriteLn('OCEAN square:')
      else
      WriteLn('TERRAIN square:');
    end;

  plotTemperature(clima, w, i, j, output);
  earthInclination := computeEarthInclination(day);
  energyIn := computeEnergyFromSunOnSquare(i, j, earthInclination, clima, world);
  if output then WriteLn('Energy from Sun entering the square: '+FloatToStr(energyIn));
  if output then WriteLn('Distributing energy between atmosphere and terrain...');
  spreadEnergyOnAtmosphereAndTerrain(clima, energyIn, i, j);
  if output then WriteLn('Temperature after insulation: ');
  plotTemperature(clima, w, i, j, output);
  if output then WriteLn('Exchanging energy between atmosphere and terrain...');
  exchangeEnergyBetweenAtmAndTerrain(clima, w, i, j);
  if output then WriteLn('Temperature after exchange: ');
  plotTemperature(clima, w, i, j, output);
  if output then WriteLn('Radiating energy back into space');
  radiateEnergyIntoSpace(clima, w, i, j);
  if output then WriteLn('Temperature after loss into space: ');
  plotTemperature(clima, w, i, j, output);
end;

procedure testLossOfEnergyDuringNight(lat, lon : TClimateType; day : Longint; var clima : TClima; var w : TWorld; output : Boolean);
var earthInclination,
    energyIn : TClimateType;
    i, j : Longint;
begin
  i := LonToX(lon);
  j := LatToY(lat);

  if output then WriteLn('Exchanging energy between atmosphere and terrain...');
  exchangeEnergyBetweenAtmAndTerrain(clima, w, i, j);
  if output then WriteLn('Temperature after exchange: ');
  plotTemperature(clima, w, i, j, output);
  if output then WriteLn('Radiating energy back into space');
  radiateEnergyIntoSpace(clima, w, i, j);
  if output then WriteLn('Temperature after loss into space: ');
  plotTemperature(clima, w, i, j, output);
end;

begin
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, 16, 16);

  WriteLn('Test of energy flow at 22 March Giubiasco (45 deg N, 6 deg E)');
  testEnergyFlowOnSquare(45, 6, 80, clima, world, true);
  WriteLn('-------------------------------------------------');

  WriteLn('Test of energy flow at 22 March Equator Ocean(0 deg S, 3 deg W)');
  testEnergyFlowOnSquare(0, -3, 80, clima, world, true);
  WriteLn('-------------------------------------------------');

  WriteLn('Test of energy flow at 22 March Equator Terrain (3 deg S, 12 deg E)');
  testEnergyFlowOnSquare(-3, 12, 80, clima, world, true);
  WriteLn('-------------------------------------------------');

  WriteLn('Test of energy flow at 22 March North pole (89 deg N, 3 deg W)');
  testEnergyFlowOnSquare(89, -3, 80, clima, world, true);
  WriteLn('-------------------------------------------------');

  WriteLn('Test of energy flow at 22 March South pole (89 deg S, 3 deg W)');
  testEnergyFlowOnSquare(-89, -3, 80, clima, world, true);
  WriteLn('-------------------------------------------------');
  WriteLn;
  WriteLn;

  WriteLn('Six months of full insulation with no nights');
  for countDay :=1 to 180 do
    for countHour := 0 to 23 do
     begin
      testEnergyFlowOnSquare(0, -3, 80+countday, clima, world, false);
      testEnergyFlowOnSquare(-3, 12, 80+countday, clima, world, false);
      testEnergyFlowOnSquare(45, 6, 80+countday, clima, world, false);
      testEnergyFlowOnSquare(89, -3, 80+countday, clima, world, false);
      testEnergyFlowOnSquare(-89, -3, 80+countday, clima, world, false);
     end;

  WriteLn('In Giubiasco:');
  plotTemperature(clima, world, lonToX(6), latToY(45), true);
  WriteLn('At equator (ocean):');
  plotTemperature(clima, world, lonToX(-3), latToY(0), true);
  WriteLn('At equator (terrain):');
  plotTemperature(clima, world, lonToX(12), latToY(-3), true);
  WriteLn('North Pole:');
  plotTemperature(clima, world, lonToX(-3), latToY(89), true);
  WriteLn('South Pole:');
  plotTemperature(clima, world, lonToX(-3), latToY(-89), true);

  WriteLn;
  WriteLn;
  WriteLn('Init world');
  initWorld(world, '..\');
  WriteLn('Init clima');
  initClima(world, clima, 16, 16);

  WriteLn('Six months of deep night without sun');
  for countDay :=1 to 180 do
    for countHour := 0 to 23 do
     begin
      testLossOfEnergyDuringNight(0, -3, 80+countday, clima, world, false);
      testLossOfEnergyDuringNight(-3, 12, 80+countday, clima, world, false);
      testLossOfEnergyDuringNight(45, 6, 80+countday, clima, world, false);
      testLossOfEnergyDuringNight(89, -3, 80+countday, clima, world, false);
      testLossOfEnergyDuringNight(-89, -3, 80+countday, clima, world, false);
     end;

  WriteLn('In Giubiasco:');
  plotTemperature(clima, world, lonToX(6), latToY(45), true);
  WriteLn('At equator (ocean):');
  plotTemperature(clima, world, lonToX(-3), latToY(0), true);
  WriteLn('At equator (terrain):');
  plotTemperature(clima, world, lonToX(12), latToY(-3), true);
  WriteLn('North Pole:');
  plotTemperature(clima, world, lonToX(-3), latToY(89), true);
  WriteLn('South Pole:');
  plotTemperature(clima, world, lonToX(-3), latToY(-89), true);

  Sleep(4000);
  ReadLn;

end.
