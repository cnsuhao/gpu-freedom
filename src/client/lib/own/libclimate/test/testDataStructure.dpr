program testDataStructure;

{$APPTYPE CONSOLE}

uses
  SysUtils, energyfunctions,conversion, datastructure,climaconstants,initmodel;

var clima : TClima;
    world : TWorld;
    j     : Longint;
    sum   : TClimateType;
begin
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, 16, 16, '');
  WriteLn;
  WriteLn('Test of data structure');
  WriteLn;
  WriteLn('Test of area of a degree squared');
  sum := 0;
  for j:=0 to 179 do
    sum := sum + world.area_of_degree_squared[j];

  sum := sum * 360;
  WriteLn('Sum of surface of degree squared across one longitude * 360 is');
  WriteLn(FloatToStr(sum));
  WriteLn('Total surface of a sphere is ');
  WriteLn(FloatToStr(4*Pi*TPhysConst.earth_radius*TPhysConst.earth_radius));
  WriteLn;

  for j:=90 downto 0 do
    WriteLn('Area of degree squared lat +'+IntToStr(j)+' :'+FloatToStr(world.area_of_degree_squared[LatToY(j)]));

  {
  WriteLn('Area of degree squared lat +89 '+FloatToStr(world.area_of_degree_squared[LatToY(89)]));
  WriteLn('Area of degree squared lat +88 '+FloatToStr(world.area_of_degree_squared[LatToY(88)]));
  WriteLn('Area of degree squared lat +01 '+FloatToStr(world.area_of_degree_squared[LatToY(1)]));
  WriteLn('Area of degree squared lat +00 '+FloatToStr(world.area_of_degree_squared[LatToY(0)]));
  WriteLn('Area of degree squared lat -01 '+FloatToStr(world.area_of_degree_squared[LatToY(-1)]));
  WriteLn('Area of degree squared lat -88 '+FloatToStr(world.area_of_degree_squared[LatToY(-88)]));
  WriteLn('Area of degree squared lat -89 '+FloatToStr(world.area_of_degree_squared[LatToY(-89)]));
  WriteLn('Area of degree squared lat -90 '+FloatToStr(world.area_of_degree_squared[LatToY(-90)]));
  }
  //TODO: move here representation of planet and save in planet.txt

  ReadLn;

end.
