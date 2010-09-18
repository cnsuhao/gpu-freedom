program testEnergyFunctions;

{$APPTYPE CONSOLE}

uses
  SysUtils, datastructure, initmodel, energyfunctions, conversion;

var clima : TClima;
    world : TWorld;

begin
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, 16, 16);


  WriteLn('Test of routines in unit energyfunctions');
  WriteLn;
  WriteLn('Earth inclination (1 January) '+FloatToStr(computeEarthInclination(1)));
  WriteLn;
  WriteLn('Earth inclination (22 March) '+FloatToStr(computeEarthInclination(80)));
  WriteLn('Earth inclination (22 June) '+FloatToStr(computeEarthInclination(172)));
  WriteLn;
  WriteLn('Earth inclination mid of year: '+FloatToStr(computeEarthInclination(182)));
  WriteLn;
  WriteLn('Earth inclination (21 September) '+FloatToStr(computeEarthInclination(264)));
  WriteLn('Earth inclination (22 December) '+FloatToStr(computeEarthInclination(356)));
  WriteLn;
  WriteLn('Earth inclination (31 December) '+FloatToStr(computeEarthInclination(365)));
  WriteLn;
  WriteLn;
  WriteLn;

  WriteLn('To compare the energies, we multiply with the area of degree squared');
  WriteLn('Energy from sun at 22 June at North Pole '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(90),23.45)));
  WriteLn('Energy from sun at 22 June at 45 deg lat N '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(45),23.45)));
  WriteLn('Energy from sun at 22 June at 23.45 deg lat N '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(23.45),23.45)));
  WriteLn('Energy from sun at 22 June at equator  '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(0),23.45)));
  WriteLn('Energy from sun at 22 June at 23.45 deg lat S '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-23.45),23.45)));
  WriteLn('Energy from sun at 22 June at 45 deg lat S '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-45),23.45)));
  WriteLn('Energy from sun at 22 June at South Pole '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-90),23.45)));

  WriteLn;
  WriteLn;
  WriteLn('Energy from sun at 22 December at North Pole '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(90),-23.45)));
  WriteLn('Energy from sun at 22 December at 45 deg lat N '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(45),-23.45)));
  WriteLn('Energy from sun at 22 December at 23.45 deg lat N '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(23.45),-23.45)));
  WriteLn('Energy from sun at 22 December at equator  '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(0),-23.45)));
  WriteLn('Energy from sun at 22 December at 23.45 deg lat S '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-23.45),-23.45)));
  WriteLn('Energy from sun at 22 December at 45 deg lat S '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-45),-23.45)));
  WriteLn('Energy from sun at 22 December at South Pole '+
           FloatToStr( computeEnergyFactorWithAngle(1,LatToY(-90),-23.45)));

  WriteLn;
  WriteLn;
  WriteLn('Energy from sun at 22 March/21 September at North Pole '+
           FloatToStr( //world.area_of_degree_squared[LatToY(90)] *
                       computeEnergyFactorWithAngle(1,LatToY(90),0)));
  WriteLn('Energy from sun at 22 March/21 September at 45 deg lat N '+
           FloatToStr( //world.area_of_degree_squared[LatToY(45)] *
                       computeEnergyFactorWithAngle(1,LatToY(45),0)));
  WriteLn('Energy from sun at 22 March/21 September at 23.45 deg lat N '+
           FloatToStr( //world.area_of_degree_squared[LatToY(23.45)] *
                       computeEnergyFactorWithAngle(1,LatToY(23.45),0)));
  WriteLn('Energy from sun at 22 March/21 September at equator  '+
           FloatToStr( //world.area_of_degree_squared[LatToY(0)] *
                       computeEnergyFactorWithAngle(1,LatToY(0),0)));
  WriteLn('Energy from sun at 22 March/21 September at 23.45 deg lat S '+
           FloatToStr( //world.area_of_degree_squared[LatToY(-23.45)] *
                       computeEnergyFactorWithAngle(1,LatToY(-23.45),0)));
  WriteLn('Energy from sun at 22 March/21 September at 45 deg lat S '+
           FloatToStr( //world.area_of_degree_squared[LatToY(-45)] *
                       computeEnergyFactorWithAngle(1,LatToY(-45),0)));
  WriteLn('Energy from sun at 22 March/21 September at South Pole '+
           FloatToStr( //world.area_of_degree_squared[LatToY(-90)] *
                       computeEnergyFactorWithAngle(1,LatToY(-90),0)));

  ReadLn;

end.
