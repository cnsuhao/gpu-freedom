program testConversion;

{$APPTYPE CONSOLE}

uses
  SysUtils, energyfunctions,conversion, datastructure, initmodel;

var clima : TClima;
    world : TWorld;

begin
  WriteLn('Init world');
  initWorld(world, '');
  WriteLn('Init clima');
  initClima(world, clima, 16, 16);

  WriteLn('Test of conversion routines');
  WriteLn;
  WriteLn('0 grad Celsius in Kelvin '+FloatToStr(CtoK(0)));
  WriteLn('-273.15 grad Celsius in Kelvin '+FloatToStr(CtoK(-273.15)));
  WriteLn;
  WriteLn('0      grad Kelvin in Celsius '+FloatToStr(KtoC(0)));
  WriteLn('273.16 grad Kelvin in Celsius '+FloatToStr(KtoC(273.16)));
  WriteLn('373.16 grad Kelvin in Celsius '+FloatToStr(KtoC(373.16)));
  WriteLn;

  WriteLn('Longitude 180 deg W on grid '+FloatToStr(LonToX(-180)));
  WriteLn('Longitude 179 deg W on grid '+FloatToStr(LonToX(-179)));
  WriteLn('Longitude 1 deg W on grid '+FloatToStr(LonToX(-1)));
  WriteLn('Longitude 0.3 deg W on grid '+FloatToStr(LonToX(-0.3)));
  WriteLn('Longitude Greenwich on grid '+FloatToStr(LonToX(0)));
  WriteLn('Longitude 0.3 deg E on grid '+FloatToStr(LonToX(0.3)));
  WriteLn('Longitude 1 deg E on grid '+FloatToStr(LonToX(1)));
  WriteLn('Longitude 179 deg E on grid '+FloatToStr(LonToX(179)));
  WriteLn('Longitude 180 deg E on grid '+FloatToStr(LonToX(180)));
  WriteLn;
  WriteLn('Latitudine North Pole on grid '+FloatToStr(LatToY(90)));
  WriteLn('Latitudine 89.5 deg on grid '+FloatToStr(LatToY(89.3)));
  WriteLn('Latitudine 89 deg on grid '+FloatToStr(LatToY(89)));
  WriteLn('Latitudine 88.3 deg on grid '+FloatToStr(LatToY(88.3)));
  WriteLn('Latitudine 88 deg on grid '+FloatToStr(LatToY(88)));
  WriteLn('Latitudine Poschiavo on grid '+FloatToStr(LatToY(45)));
  WriteLn('Latitudine 0.3 deg on grid '+FloatToStr(LatToY(0.3)));
  WriteLn('Latitudine equator on grid '+FloatToStr(LatToY(0)));
  WriteLn('Latitudine -0.3 deg on grid '+FloatToStr(LatToY(-0.3)));
  WriteLn('Latitudine -88 deg on grid '+FloatToStr(LatToY(-88)));
  WriteLn('Latitudine -89 deg on grid '+FloatToStr(LatToY(-89)));
  WriteLn('Latitudine -89.5 deg on grid '+FloatToStr(LatToY(-89.3)));
  WriteLn('Latitudine south pole on grid '+FloatToStr(LatToY(-90)));
  WriteLn;
  WriteLn;
  WriteLn('Converting twice 90 deg lat N : '+FloatToStr(YtoLat(LatToY(90))));
  WriteLn('Converting twice 0 deg lat : '+FloatToStr(YtoLat(LatToY(0))));
  WriteLn('Converting twice 90 deg lat S: '+FloatToStr(YtoLat(LatToY(-90))));
  WriteLn;
  WriteLn('Converting twice 180 deg lat W : '+FloatToStr(XtoLon(LonToX(-180))));
  WriteLn('Converting twice 0 deg lat : '+FloatToStr(XtoLon(LonToX(0))));
  WriteLn('Converting twice 180 deg lat E: '+FloatToStr(XtoLon(LonToX(180))));

  ReadLn;

end.
