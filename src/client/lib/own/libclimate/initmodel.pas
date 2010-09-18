unit initmodel;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, datastructure, climaconstants, conversion, statechanges, co2cycle;

  procedure initTimeHDY(hour, day, year : Longint; var t : TTime; var sSurface : TSolarSurface);
  procedure initTime(var t : TTime; var sSurface : TSolarSurface);
  procedure stepTime(var t : TTime; var sSurface : TSolarSurface);

  procedure initWorld(var w : TWorld; filePath : String);
  procedure initClima(var w : TWorld; var clima : TClima; T_atm, T_ocean_terr : TClimateType;
                      filePath : String);


implementation

function ExtractParam(var S: string; Separator: string): string;
var
  i: Longint;
begin
  i := Pos(Separator, S);
  if i > 0 then
  begin
    Result := Copy(S, 1, i - 1);
    Delete(S, 1, i);
  end
  else
  begin
    Result := S;
    S      := '';
  end;
end;

procedure initTime(var t : TTime; var sSurface : TSolarSurface);
begin
     initTimeHDY(0, 80, 2000, t, sSurface);
end;

procedure initTimeHDY(hour, day, year : Longint; var t : TTime; var sSurface : TSolarSurface);
begin
   if (hour<0) or (hour>23) then raise Exception.create('Hour has to lie between 0 and 23');
   if (day<1) or (day>365) then raise Exception.create('Day has to lie between 1 and 365');
   if (year<0) or (year>10000) then raise Exception.create('Year has to lie between 0 and 10000');

   t.hour := hour;
   t.day := day;
   t.year := year;

   sSurface.degstart := TMdlConst.initDegreeSunlight+hour;
   sSurface.degend := TMdlConst.initDegreeSunlight+(hour+12)*15; // 12 hours which are 15 degrees a part
   if sSurface.degstart >= 360 then sSurface.degstart :=  sSurface.degstart - 360;
   if sSurface.degend >= 360 then sSurface.degend := sSurface.degend - 360;
end;

procedure stepTime(var t : TTime; var sSurface : TSolarSurface);
begin
  t.hour := t.hour + TSimConst.degree_step/15;
  if t.hour>=24 then
    begin
      t.hour := 0;
      t.day := t.day + 1;
      if (t.day>365) then
         begin
            t.day := 1;
            t.year := t.year + 1;
         end;
    end;

    if TMdlConst.rotation then
         begin
           sSurface.degstart := sSurface.degstart-TSimConst.degree_step*TMdlConst.inverse_rotation;
           if sSurface.degstart <0 then sSurface.degstart :=  sSurface.degstart + 360;
           if sSurface.degstart >=360 then sSurface.degstart :=  sSurface.degstart - 360;

           sSurface.degend := sSurface.degstart-180*TMdlConst.inverse_rotation;
           if sSurface.degend <0 then sSurface.degend := sSurface.degend + 360;
           if sSurface.degend >=360 then sSurface.degend := sSurface.degend - 360;
         end;
end;


procedure initWorld(var w : TWorld; filePath : String);
var G : TextFile;
    i, j : Longint;
    str, AppPath : String;

begin
  initModelParameters;
  initConversion(false); // default to linear grid

  if (filePath='') then AppPath := ExtractFilePath(ParamStr(0))
           else AppPath := filePath;

  // load elevations
  AssignFile(G, AppPath+'config\thedayaftertomorrow\planet-elevation.txt');
 try
  Reset(G);
  for j := 0 to 179  do
   begin
      for i:=0 to 359 do
       begin
             ReadLn(G, str);
             if (Trim(str)<>'') then
               begin
                w.elevation[i][j] := StrToFloat(str);
                w.isOcean[i][j] := w.elevation[i][j] <=0;
               end
              else
               raise Exception.Create('Problem in file planet-elevation.txt');
       end;
   end;
  finally
   Closefile(G);
  end;
end;

procedure initClima(var w : TWorld; var clima : TClima; T_atm, T_ocean_terr : TClimateType; filePath : String);
var i, j, l,
    x, y, population : Longint;
    kT_atm,
    kT_ocean_terr,
    weight,
    altitude,
    lat,
    lon  : TClimateType;

    h, hStart, hEnd,
    startLat, stopLat,
    area : TClimateType;

    AppPath, str : String;
    G : Textfile;
begin
  if (filePath='') then AppPath := ExtractFilePath(ParamStr(0))
           else AppPath := filePath;

// load population
 for j := 0 to 179  do
    for i:=0 to 359 do
       clima.population[i][j]:=0;
try
  AssignFile(G, AppPath+'config\thedayaftertomorrow\planet-population-1990.txt');
  Reset(G);
  while not Eof(G) do
       begin
             ReadLn(G, str);
             if (Trim(str)<>'') then
               begin
                 // parse string
                 y := LatToY(90-StrToInt(ExtractParam(str,';')));
                 x := LonToX(StrToInt(ExtractParam(str,';'))-180);
                 population := StrToInt(str);
                 clima.population[x][y]:=clima.population[x][y]+population;
               end;
       end;
  finally
   Closefile(G);
  end;
  increasePopulation(clima, 3650); // the dataset is of 1990, simulation begins in 2000

   // temperature
   { this initialization is very simplistic
    we assume the earth has only one temperature in atmosphere and one in terrain
    and distribute the energy over ocean, terrain, atmosphere and ice
   }
   // compute the area of a degree squared
   // a zone on a sphere has area 2*Pi*r*h
   // we divide the zone in 360 degrees
   for j:=0 to 179 do
      begin
         startLat := Abs(YtoLat(j));
         stopLat := Abs(YtoLat(j+1));

         hStart := cos((90-startLat)/360*2*Pi) * TPhysConst.earth_radius;
         hEnd   := cos((90-stopLat)/360*2*Pi) * TPhysConst.earth_radius;

         h := Abs(hStart - hEnd);
         area := 2*Pi*TPhysConst.earth_radius*h;
         w.area_of_degree_squared[j]:=area/360;
         w.length_of_degree[j] := Sqrt(w.area_of_degree_squared[j]);
      end;

  for j := 0 to 179  do
    for i:=0 to 359 do
        begin
          lat := YtoLat(j);
          lon := XtoLon(i);

          clima.T_ocean_terr[i][j] := 15.5+TPhysConst.absolutezero;
          clima.T_atmosphere[0][i][j] := 15.5+TPhysConst.absolutezero;


          kT_atm := CtoK(T_atm+TInitCond.thermic_excursion* cos(Abs(lat)/90*Pi/2) );
          kT_ocean_terr:= CtoK(T_ocean_terr+ TInitCond.thermic_excursion * cos(Abs(lat)/90*Pi/2) );

          // desert belt
          if (Abs(lat)<=TInitCond.desert_belt_lat+TInitCond.desert_belt_ext) and
                 (Abs(lat)>=TInitCond.desert_belt_lat) and (not w.isOcean[i][j]) then
                     kT_ocean_terr := kT_ocean_terr + TInitCond.desert_belt_delta_T * sin(Abs(TInitCond.desert_belt_lat-lat/(TInitCond.desert_belt_ext))*Pi);


          if (not w.isOcean[i][j]) then
            begin
              if (Abs(lat)<60) then kT_ocean_terr := kT_ocean_terr + TInitCond.surface_shift;

              // thermic gradient adjustment on surface and atmosphere
              kT_ocean_terr := kT_ocean_terr - TInitCond.thermic_gradient_avg * Abs(w.elevation[i][j]/100);
              kT_atm := kT_atm - TInitCond.thermic_gradient_avg * Abs(w.elevation[i][j]/100);

            end
          else
            begin
             // thermic gradient adjustment on sea depth and atmopshere
             kT_ocean_terr := kT_ocean_terr - TInitCond.thermic_gradient_sea * Abs(w.elevation[i][j]/1000);
             kT_atm := kT_atm - TInitCond.thermic_gradient_sea * Abs(w.elevation[i][j]/1000);
            end;

          // insulation
          if (lon>-180) and (lon<=0) then
             begin
               // heat terrain up and cool ocean down
               if (w.isOcean[i][j]) then
                  kT_ocean_terr := kT_ocean_terr-TInitCond.ocean_shift
               else
                  kT_ocean_terr := kT_ocean_terr+TInitCond.terrain_shift;

             end
          else
             begin
               // heat ocean up, cool terrain down
               if (w.isOcean[i][j]) then
                  kT_ocean_terr := kT_ocean_terr+TInitCond.ocean_shift
               else
                  kT_ocean_terr := kT_ocean_terr-TInitCond.terrain_shift;

             end;
          clima.T_ocean_terr[i][j] := kT_ocean_terr;
          clima.T_atmosphere[0][i][j] := kT_atm;

          // steam and rain
          for l:=0 to TMdlConst.atmospheric_layers-1 do
             clima.steam[l][i][j] := 0;
          clima.rain[i][j] := false;
          clima.water_surface[i][j] := 0;
          clima.rain_times[i][j] := 0;

          clima.humidity[i][j] := 0;
          clima.rain_times[i][j] := 0;

          // ashes and CO2
          clima.ashes_pct[i][j] := 0;
          clima.co2_tons[i][j] := 0;

          // atmospheric layers
          for l:=1 to TMdlConst.atmospheric_layers-1 do
             clima.T_atmosphere[l][i][j] := thermicGradient(w, clima, l*TMdlConst.distance_atm_layers, clima.T_atmosphere[0][i][j], i, j);


          clima.isIce[i][j] := (clima.T_ocean_terr[i][j]<=TPhysConst.kT_Ice);


          // update energies
          // deltaQ = cp * m * deltaT
          if w.isOcean[i][j] then
              altitude := 0
          else
              altitude := w.elevation[i][j];

          // energies
          weight := weightOnAltitudeProQuadrateMeter(altitude, i, j, w);
          clima.energy_atmosphere[i][j] := clima.T_atmosphere[0][i][j] * TPhysConst.cp_air * weight * w.area_of_degree_squared[j];
          // terrain
          clima.energy_ocean_terr[i][j] := clima.T_ocean_terr[i][j] * TPhysConst.cp_earth * (w.elevation[i][j]+TSimConst.earth_crust_height) * w.area_of_degree_squared[j] * TPhysConst.density_earth;

          if w.isOcean[i][j] then
                 clima.energy_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] + clima.T_ocean_terr[i][j] * TPhysConst.cp_water * Abs(w.elevation[i][j]) * w.area_of_degree_squared[j] * TPhysConst.density_water;

        end;
end;

end.

