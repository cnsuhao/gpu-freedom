unit earthDrawingControl;

{$mode objfpc}{$H+}
interface

uses
  Classes, SysUtils, Controls, Graphics, LCLType, datastructure,
  conversion, riverandlakes, statechanges, climaconstants, Dialogs,
  energyfunctions;


const
  PAINT_EARTH_ATMOSPHERE         = 100;
  PAINT_WIND                     = 200;
  PAINT_MARINE_CURRENTS          = 300;
  PAINT_TEMPERATURE_SURFACE      = 400;
  PAINT_TEMPERATURE_ATM          = 500;
  PAINT_WATER_AND_VEGETATION     = 600;
  PAINT_WATER                    = 700;
  PAINT_CLOUDS                   = 800;
  PAINT_SURFACE_TRANSFER         = 900;
  PAINT_ASHES                    = 1000;
  PAINT_CO2                      = 1100;

  // colors taken from
  // http://www.lohninger.com/packages.html
  // SDL Delphi Components (c) by Lohninger
  // please use them only for educational purposes
  clLightSkyBlue         = TColor($00FACE87);
  clLightBlue            = TColor($00E6D8AD);
  clDarkBlue             = TColor($008B0000);
  clDarkGray             = TColor($00A9A9A9);
  clAntiqueWhite         = TColor($00D7EBFA);
  clBeige                = TColor($00DCF5F5);
  clLightYellow          = TColor($00E0FFFF);
  clLightGray            = TColor($00D3D3D3);
  clOrange               = TColor($0000A5FF);
  clBrown                = TColor($002A2AA5);
  clLightGreen           = TColor($0090EE90);
  clAzure                = TColor($00FFFFF0);
  clAquamarine           = TColor($00D4FF7F);
  clYellowGreen          = TColor($0032CD9A);
  clForestGreen          = TColor($00228B22);




type
  TEarthDrawingControl = class(TCustomControl)
  public
    constructor Create(obj : TComponent; w : PWorld; clima : PClima; ss : PSolarSurface; tt :PTime; PLOT_MODE : Longint; size : Longint);
    procedure Paint; override;
    procedure paintEarth(var w : TWorld; var clima : TClima; var ts : TSolarSurface); overload;
    procedure setPlotMode(plotmode : Longint);
    function  getPlotMode : Longint;
    function  getColors : PGridColor;

    procedure MouseDown(Button: TMouseButton; Shift: TShiftState; X, Y: Integer); override;


  private
    procedure paintEarth(var w : TWorld; var clima : TClima; var ts : TSolarSurface; PLOT_MODE : Longint); overload;
    procedure paintEarthAndClouds(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint);
    procedure paintTemperature(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
    procedure paintWindorCurrentOrSurface(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
    procedure paintWaterAndVegetation(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
    procedure paintWater(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
    procedure paintTerrain(var w : TWorld; var clima : TClima; i, j : Longint);
    procedure paintClouds(var w : TWorld; var clima : TClima; i, j : Longint);
    procedure paintAshes(var w : TWorld; var clima : TClima; i, j : Longint);
    procedure paintCO2(var w : TWorld; var clima : TClima; i, j : Longint);
    procedure paintCoasts(var w : TWorld; var clima : TClima; i, j : Longint);
    function countClouds(var w : TWorld; var clima : TClima; i, j : Longint) : Longint;
    function countSteam(var w : TWorld; var clima : TClima; i, j : Longint) : TClimateType;
    colors : TGridColor;

    _size : Longint; //1,2,3,4
    _plotMode : Longint;
    _clima : PClima;
    _world : PWorld;
    _ss : PSolarSurface;
    _tt : PTime;
  end;

  type PEarthDrawingControl = ^TEarthDrawingControl;

implementation


constructor TEarthDrawingControl.Create(obj : TComponent; w : PWorld; clima : PClima; ss : PSolarSurface; tt : PTime; PLOT_MODE : Longint; size : Longint);
begin
  inherited Create(obj);
  _size := size;
  _plotMode := PLOT_MODE;
  _clima := clima;
  _world := w;
  _ss := ss;
  _tt := tt;
  Height := _size*180+1;
  Width := _size*360+1;
  paintEarth(w^, clima^, ss^);
end;

procedure TEarthDrawingControl.setPlotMode(plotmode : Longint);
begin
 _plotMode := plotMode;
end;

function TEarthDrawingControl.getPlotMode : Longint;
begin
 Result := _plotMode;
end;

function TEarthDrawingControl.getColors : PGridColor;
begin
  Result := @colors;
end;


procedure TEarthDrawingControl.Paint;
var
  i, j: Integer;
  Bitmap: TBitmap;
begin
  Bitmap := TBitmap.Create;
  try
    Bitmap.Height := Height;
    Bitmap.Width := Width;

    for j := 0 to 179 do
     for i := 0 to 359 do
       begin
         Bitmap.Canvas.Pen.Color := colors[i][j];
         Bitmap.Canvas.Brush.Color := colors[i][j];
         if (_size>1) then
           Bitmap.Canvas.Rectangle(i*_size ,j*_size, i*_size+_size, j*_size+_size)
         else
          begin
           Bitmap.Canvas.MoveTo(i, j);
           Bitmap.Canvas.LineTo(i+1, j+1);
          end;
       end;

    Canvas.Draw(0, 0, Bitmap);
  finally
    Bitmap.Free;
  end;

  inherited Paint;
end;


procedure TEarthDrawingControl.paintEarth(var w : TWorld; var clima : TClima; var ts : TSolarSurface);
begin
  paintEarth(w, clima, ts, _plotMode);
end;


procedure TEarthDrawingControl.paintEarth(var w : TWorld; var clima : TClima; var ts : TSolarSurface; PLOT_MODE : Longint);
var
  i, j: Integer;
begin
    for j := 0 to 179 do
     for i := 0 to 359 do
       begin
         if plot_Mode = PAINT_EARTH_ATMOSPHERE then
            paintEarthAndClouds(w, clima, ts, i, j)
         else
         if ((plot_Mode = PAINT_TEMPERATURE_SURFACE)
             or
            (plot_Mode = PAINT_TEMPERATURE_ATM)) then
            paintTemperature(w, clima, ts, i, j, PLOT_MODE)
         else
         if (plot_Mode = PAINT_WIND) then
            paintWindorCurrentOrSurface(w, clima, ts, i, j, PAINT_WIND)
         else
         if (plot_Mode = PAINT_MARINE_CURRENTS) then
            paintWindorCurrentOrSurface(w, clima, ts, i, j, PAINT_MARINE_CURRENTS)
         else
         if (plot_Mode = PAINT_SURFACE_TRANSFER) then
            paintWindorCurrentOrSurface(w, clima, ts, i, j, PAINT_SURFACE_TRANSFER)
         else
         if (plot_Mode = PAINT_WATER_AND_VEGETATION) then
            paintWaterAndVegetation(w, clima, ts, i, j, PAINT_WATER_AND_VEGETATION)
         else
         if (plot_Mode = PAINT_WATER) then
            paintWater(w, clima, ts, i, j, PAINT_WATER)
         else
         if (plot_Mode = PAINT_CLOUDS) then
            paintClouds(w, clima, i, j)
         else
         if (plot_Mode = PAINT_ASHES) then
            paintAshes(w, clima, i, j)
         else
         if (plot_Mode = PAINT_CO2) then
            paintCO2(w, clima, i, j)
         else
           raise Exception.Create('Unknown PlotMode');
       end;
end;

procedure  TEarthDrawingControl.paintEarthAndClouds(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint);
var
   _isinSunlight : Boolean;
   _TSurface : TClimateType;
begin
          _TSurface := KToC(clima.T_ocean_terr[i][j]);
          _isinSunlight := isInSunlight(i, j, ts);
          if (w.isOcean[i][j]) then
            begin
              if _isInSunlight then
                begin
                 if (w.elevation[i][j]<-5000) then
                  colors[i][j] := clBlue
                 else
                 if (w.elevation[i][j]<-3000) then
                  colors[i][j] := clLightSkyBlue
                 else
                  colors[i][j] := clLightBlue;
                  if clima.isIce[i][j] then colors[i][j] := clAntiqueWhite;
                end
              else
                begin
                 colors[i][j] := clDarkBlue;
                 if clima.isIce[i][j] then colors[i][j] := clDarkGray;
                end;

            end
          else
           begin
             if _isInSunlight then
                 begin
                   if w.elevation[i][j]<1800 then
                       begin
                         if (_TSurface<7) then
                              colors[i][j] := clLightYellow
                         else
                         if (_TSurface<40) then
                              colors[i][j] := clBeige
                         else
                         if (_TSurface<50) then
                              colors[i][j] := clBrown
                         else
                              colors[i][j] := clOrange
                       end
                     else
                       colors[i][j] := clLightGray;

                     paintTerrain(w, clima, i, j);
                 end
              else
                begin
                 colors[i][j] := clBlack;
                 if clima.isIce[i][j] then colors[i][j] := clDarkGray;

                 if TSimConst.population and (clima.population[i][j]>1.5E6) then colors[i][j] := clYellow;
                end;
           end;


          //if countSteam(w, clima, i, j)>= TSimConst.paint_clouds then
          if countClouds(w, clima, i, j)>=4 then
             if _isInSunlight then
                colors[i][j] := clWhite
               else
                colors[i][j] := clDarkGray;

        if clima.ashes_pct[i][j]>0.1 then
             colors[i][j] := clDarkGray;
end;

procedure  TEarthDrawingControl.paintTemperature(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
var temp : TClimateType;
begin
 if (PLOT_Mode = PAINT_TEMPERATURE_SURFACE) then
   temp := KtoC(clima.T_ocean_terr[i][j])
 else
   temp := KtoC(clima.T_atmosphere[0][i][j]);

 if (temp>35) then colors[i][j] := clRed
 else
 if (temp>25) then colors[i][j] := clOrange
 else
 if (temp>16) then colors[i][j] := clYellow
 else
 if (temp>10) then colors[i][j] := clGreen
 else
 if (temp>5) then colors[i][j] := clBrown
 else
 if (temp>0) then colors[i][j] := clBlack
 else colors[i][j] := clWhite;
end;

procedure  TEarthDrawingControl.paintWindorCurrentorSurface(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
var direction : ShortInt;
begin
  colors[i][j] := clBeige;

  if (PLOT_MODE = PAINT_WIND) then
     direction := clima.wind[0][i][j]
  else
  if (PLOT_MODE = PAINT_MARINE_CURRENTS) and (w.isOcean[i][j] and (not clima.isIce[i][j])) then
     direction := clima.surfaceTransfer[i][j]
  else
  if (PLOT_MODE = PAINT_SURFACE_TRANSFER) and ((not w.isOcean[i][j]) or clima.isIce[i][j]) then
     direction := clima.surfaceTransfer[i][j]
  else
     Exit;

  if direction = NONE then
     colors[i][j] := clWhite
  else
  if direction = NORTH_WEST then
     colors[i][j] := clRed
  else
  if direction = NORTH then
     colors[i][j] := clOrange
  else
  if direction = NORTH_EAST then
     colors[i][j] := clYellow
  else
  if direction = WEST then
     colors[i][j] := clGreen
  else
  if direction = EAST then
     colors[i][j] := clLightGreen
  else
  if direction = SOUTH_WEST then
     colors[i][j] := clBlue
  else
  if direction = SOUTH then
     colors[i][j] := clAzure
  else
  if direction = SOUTH_EAST then
     colors[i][j] := clAquamarine;

end;

procedure TEarthDrawingControl.paintTerrain(var w : TWorld; var clima : TClima; i, j : Longint);
begin
       if clima.isIce[i][j] then colors[i][j] := clAntiqueWhite
       else
       if isDesert(w, clima, i, j) then colors[i][j] := clYellow
       else
       if isJungle(w, clima, i, j) then colors[i][j] := clGreen
       else
       if isForest(w, clima, i, j) then colors[i][j] := clForestGreen
       else
       if isRiver(clima, i, j) then colors[i][j] := clBlue
       else
       if isGrass(w, clima, i, j) then colors[i][j] := clYellowGreen;
end;


procedure TEarthDrawingControl.paintWaterAndVegetation(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
begin
 colors[i][j] := clBeige;

 if w.IsOcean[i][j] then
    colors[i][j] := clAzure
  else
    paintTerrain(w, clima, i, j);

 if clima.IsIce[i][j] then
    colors[i][j] := clWhite;

end;

procedure TEarthDrawingControl.paintWater(var w : TWorld; var clima : TClima; var ts : TSolarSurface; i, j : Longint; PLOT_MODE : Longint);
var _TSurface : TClimateType;
begin
 colors[i][j] := clGray;

 if w.IsOcean[i][j] then
   begin
    colors[i][j] := clAzure;
     if clima.IsIce[i][j] then colors[i][j] := clRed;
   end
  else
    begin
      if clima.water_surface[i][j]>0 then colors[i][j] := clLightBlue;
      if (clima.water_surface[i][j]>0.5*clima.avgWaterSurface[j]) then colors[i][j] := clBlue;
      if clima.IsIce[i][j] then colors[i][j] := clRed;
    end;

end;


procedure TEarthDrawingControl.paintClouds(var w : TWorld; var clima : TClima; i, j : Longint);
var count : Longint;
begin
 paintCoasts(w, clima, i, j);
 count := countClouds(w, clima, i, j);

 if count = 1 then colors[i][j]:=clLightYellow
 else
 if count = 2 then colors[i][j]:=clYellow
 else
 if count = 3 then colors[i][j]:=clOrange
 else
 if count = 4 then colors[i][j]:=clRed
 else
 if count >=5 then colors[i][j]:=clPurple;
end;

procedure TEarthDrawingControl.paintAshes(var w : TWorld; var clima : TClima; i, j : Longint);
begin
 paintCoasts(w, clima, i, j);

 if clima.ashes_pct[i][j]>0.01 then
         colors[i][j] := clRed
 else
 if clima.ashes_pct[i][j]>0.001 then
         colors[i][j] := clYellow;
end;

procedure TEarthDrawingControl.paintCO2(var w : TWorld; var clima : TClima; i, j : Longint);
begin
 paintCoasts(w, clima, i, j);

 if clima.co2_tons[i][j]>1E8 then colors[i][j] := clPurple
 else
 if clima.co2_tons[i][j]>1E7 then colors[i][j] := clRed
 else
 if clima.co2_tons[i][j]>1E6 then colors[i][j] := clOrange
 else
 if clima.co2_tons[i][j]>1E3 then colors[i][j] := clYellow
 else
 if clima.co2_tons[i][j]>1 then colors[i][j] := clLightYellow;
end;


procedure TEarthDrawingControl.paintCoasts(var w : TWorld; var clima : TClima; i, j : Longint);
begin
 colors[i][j] := clBeige;

 if w.IsOcean[i][j] then
   begin
    colors[i][j] := clLightBlue;
     if clima.IsIce[i][j] then colors[i][j] := clWhite;
   end
  else
    begin
      if clima.IsIce[i][j] then colors[i][j] := clWhite;
    end;
end;

function TEarthDrawingControl.countSteam(var w : TWorld; var clima : TClima; i, j : Longint) : TClimateType;
var l : Longint;
    steam : TClimateType;
begin
   steam := 0;
   for l:=0 to TMdlConst.atmospheric_layers-1 do
      begin
        steam := steam + clima.steam[l][i][j]/w.area_of_degree_squared[j];
      end;
  Result := steam;
end;


function TEarthDrawingControl.countClouds(var w : TWorld; var clima : TClima; i, j : Longint) : Longint;
var count, l : Longint;
begin
  count := 0;
   for l:=0 to TMdlConst.atmospheric_layers-1 do
      begin
        if (clima.steam[l][i][j]/w.area_of_degree_squared[j])>(TSimConst.paint_clouds/TMdlConst.atmospheric_layers) then Inc(count);
      end;
  Result := count;
end;

procedure TEarthDrawingControl.MouseDown(Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
var str : String;
    i, j, l : Longint;
    earthInclination, energyFactor, energyIn, energyOut : TClimateType;

      function DirToStr(dir : ShortInt) : String;
      begin
         case dir of
            NONE  : Result := 'NONE';
            NORTH : Result := 'NORTH';
            SOUTH : Result := 'SOUTH';
            EAST  : Result := 'EAST';
            WEST  : Result := 'WEST';
            NORTH_EAST : Result := 'NORTH_EAST';
            NORTH_WEST : Result := 'NORTH_WEST';
            SOUTH_EAST : Result := 'SOUTH_EAST';
            SOUTH_WEST : Result := 'SOUTH_WEST';
         end;
      end;
begin
  inherited MouseDown(Button, Shift,X, Y);
  i := X div _size; j:= Y div _size;
  Str := 'i: '+IntToStr(i)+' j: '+IntToStr(j)+#13#10;
  Str := Str + 'Longitude: '+FloatToStr(XToLon(i))+' Latitude: '+FloatToStr(YtoLat(j))+#13#10;
  Str := Str + #13#10;
  Str := Str + 'Elevation: '+ FloatToStr(_world^.elevation[i][j])+' m';
  Str := Str + #13#10;
  if isInSunlight(i, j, _ss^) then
    begin
      earthInclination := computeEarthInclination(_tt^.day);
      Str := Str + #13#10;
      Str := Str + 'Earth inclination (seasonal): '+ FloatToStr(earthInclination)+' degrees';
      energyFactor := computeEnergyFactorWithAngle(i, j, earthInclination);
      Str := Str + #13#10;
      Str := Str + 'Energy factor (%): '+ FloatToStr(energyFactor);
      Str := Str + #13#10;
      energyIn := computeEnergyFromSunOnSquare(i, j, earthInclination, _clima^, _world^);
      Str := Str + 'Incoming energy from sun: '+ FloatToStr(energyIn/_world^.area_of_degree_squared[j]/TSimConst.hour_step)+' W/m^2';
      Str := Str + #13#10;
    end;
  energyOut := computeRadiatedEnergyIntoSpace(_clima^, _world^, i, j);
  Str := Str + 'Outgoing energy into space: '+ FloatToStr(energyOut/_world^.area_of_degree_squared[j]/TSimConst.hour_step)+' W/m^2';
  Str := Str + #13#10;

  Str := Str + 'Type: ';
  if _world^.isOcean[i][j] then
     Str := Str + 'Ocean'
    else
     begin
      Str := Str + 'Terrain';
      if isDesert(_world^, _clima^, i, j) then  Str := Str + ', desert'
      else
      if isJungle(_world^, _clima^, i, j) then  Str := Str + ', jungle'
      else
      if isForest(_world^, _clima^, i, j) then  Str := Str + ', forest'
      else
      if isRiver(_clima^, i, j) then Str := Str + ', river'
      else
      if isGrass(_world^, _clima^, i, j) then Str := Str + ', grass';
     end;
  if _clima^.isIce[i][j] then
     Str := Str + ', ice covered';
  Str := Str + #13#10;
  if _clima^.population[i][j]>0 then
     Str := Str + 'Population: '+ FloatToStr(_clima^.population[i][j]);
  Str := Str + #13#10;
  Str := Str + #13#10;
  Str := Str + 'Temperatures (Celsius)'+#13#10;
  Str := Str + 'T. surface: '+FloatToStr(KtoC(_clima^.T_ocean_terr[i][j]))+#13#10;
  Str := Str + 'T. atmosphere: '+FloatToStr(KtoC(_clima^.T_atmosphere[0][i][j]))+#13#10;
  Str := Str + 'T. height: (height -> +'+IntToStr(TMdlConst.distance_atm_layers)+' m)'+#13#10;
  for l:=1 to TMdlConst.atmospheric_layers-1 do
      Str := Str + FloatToStr(KtoC(_clima^.T_atmosphere[l][i][j]))+' ';
  Str := Str + #13#10;
  Str := Str + #13#10;
  Str := Str + 'Energies (J)'+#13#10;
  Str := Str + 'E. surface: '+FloatToStr(KtoC(_clima^.energy_ocean_terr[i][j]))+#13#10;
  Str := Str + 'E. atmosphere: '+FloatToStr(KtoC(_clima^.energy_atmosphere[i][j]))+#13#10;
  Str := Str + #13#10;
  Str := Str + 'Winds (height -> +'+IntToStr(TMdlConst.distance_atm_layers)+' m)'+#13#10;
  for l:=0 to TMdlConst.atmospheric_layers-1 do
      Str := Str + DirToStr(_clima^.wind[l][i][j])+' ';
  Str := Str + #13#10;
  if _world^.isOcean[i][j] then
     Str := Str + 'Marine Current: '
    else
     Str := Str + 'Surface transfer: ';
  Str := Str + DirToStr(_clima^.surfaceTransfer[i][j])+#13#10;

  Str := Str + #13#10;
  Str := Str + 'Steam: (kg, height -> +'+IntToStr(TMdlConst.distance_atm_layers)+' m)'+#13#10;
  for l:=0 to TMdlConst.atmospheric_layers-1 do
      Str := Str + FloatToStr(_clima^.steam[l][i][j])+' ';
  Str := Str + #13#10;
  if _clima^.rain[i][j] then
    Str := Str + 'Currently raining... '+ #13#10;
  Str := Str + 'Rain times: '+FloatToStr(_clima^.rain_times[i][j])+ #13#10;;
  Str := Str + 'Humidity: '+FloatToStr(_clima^.humidity[i][j])+ #13#10;;
  Str := Str + 'Water on surface: '+FloatToStr(_clima^.water_surface[i][j])+' kg'+#13#10;;
  Str := Str + #13#10;
  Str := Str + #13#10;
  Str := Str + 'Ashes (%): '+FloatToStr(_clima^.ashes_pct[i][j])+ #13#10;;
  Str := Str + 'CO2 (tons): '+FloatToStr(_clima^.co2_tons[i][j])+ #13#10;;
  ShowMessage(Str);
end;

end.

