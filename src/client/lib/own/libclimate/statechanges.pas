unit statechanges;
{(c) 2010 HB9TVM    Source code is under GPL}

interface
uses datastructure, climaconstants, conversion, Sysutils, Math;

 function isInSunlight(i, j : Longint; var sSurface : TSolarSurface) : Boolean;

 procedure WaterOrIce(var clima : TClima; var w : TWorld; i, j : Longint);
 function evaporatePercentage(var clima : TClima;  T_param : TClimateType; i, j : Longint) : TClimateType;
 function weightOnAltitudeProQuadrateMeter(altitude : TClimateType; i,j: Longint; var w : TWorld) : TClimateType;
 function thermicGradient(var w : TWorld; var clima :TClima; elevation, T_initial : TClimateType; i,j : Longint) : TClimateType;
 function computeHumidity(var clima : TClima; var w : TWorld; i, j : Longint) :TClimateType;
 function maxWaterInSteam(var clima : TClima; var w : TWorld; l, i, j : Longint) : TClimateType;


implementation

procedure WaterOrIce(var clima : TClima; var w : TWorld; i, j : Longint);
begin
    clima.isIce[i][j] := (clima.T_ocean_terr[i][j]<=TPhysConst.kT_ice);
    //if not w.isOcean[i][j] then clima.isIce[i][j] := clima.isIce[i][j] and (clima.water_surface[i][j]>0);
end;


function evaporatePercentage(var clima : TClima; T_param : TClimateType; i, j : Longint) : TClimateType;
var T : TClimateType;
begin
  // a simple linear relation between 10 and 100 degree
  T := KtoC(T_param);
  if (T<=TSimConst.evaporation_start_temp) then Result := 0
  else
  if (T>=TSimConst.full_evaporation_temp) then Result := 1
  else
    Result := (T-TSimConst.evaporation_start_temp)/(TSimConst.full_evaporation_temp-TSimConst.evaporation_start_temp);
end;

function isInSunlight(i, j : Longint; var sSurface : TSolarSurface) : Boolean;
begin

 if sSurface.degend>sSurface.degstart then
  Result := (i>=sSurface.degstart) and (i<=sSurface.degend)
 else
  Result := (i<=sSurface.degend) or (i>=sSurface.degstart);

end;



function weightOnAltitudeProQuadrateMeter(altitude : TClimateType; i,j: Longint; var w : TWorld) : TClimateType;
var p : TClimateType;
begin
  if (altitude<0) then raise Exception.Create('weightOnAltitude(...) called with negative altitude');
  p := 100 * Math.Power((44331.514-altitude)  / 11880.516, 1/0.1902632);
  Result := p/TPhysConst.grav_acc;
end;

function thermicGradient(var w : TWorld; var clima :TClima; elevation, T_initial : TClimateType; i,j : Longint) : TClimateType;
var altitude : TClimateType;
begin
 if (elevation<=0) then Result := T_initial;
 if (elevation>TSimConst.max_atmosphere_height) then elevation := TSimConst.max_atmosphere_height;
 if clima.humidity[i][j]<0 then raise Exception.create('Humidity is negative in thermicGradient(...)');
 Result := T_initial - Abs(elevation)/100 * (1-clima.humidity[i][j]*(TSimConst.thermic_gradient_dry-TSimConst.thermic_gradient_wet));

 // outer atmospheric layers might reach zero absolute
 if (Result<0) then Result := 0;
end;

function computeHumidity(var clima : TClima; var w : TWorld; i, j : Longint) : TClimateType;
var maxWater : TClimateType;
    l        : Longint;
begin
         maxWater := 0;
         clima.humidity[i][j] := 0;
         for l:=0 to TMdlConst.atmospheric_layers-1 do
           begin
             clima.humidity[i][j] := clima.humidity[i][j]+clima.steam[l][i][j];
             maxWater := maxWater + maxWaterInSteam(clima, w, l, i, j);
           end;

         clima.humidity[i][j] := clima.humidity[i][j]/maxWater;
         if (clima.humidity[i][j] > 1) then clima.humidity[i][j] := 1;

         Result := clima.humidity[i][j];
end;



function maxWaterInSteam(var clima : TClima; var w : TWorld; l, i, j : Longint) : TClimateType;
var
    altitude,
    tCelsius,
    density  : TClimateType;
begin
 tCelsius := KtoC(clima.T_atmosphere[l][i][j]);

//Formeln und Tafeln pg 174
if clima.isIce[i][j] then
      begin
        if (tCelsius<-25) then density := 0.00035 else
        if (tCelsius<-20) then density := 0.00057 else
        if (tCelsius<-15) then density := 0.00091 else
        if (tCelsius<-10) then density := 0.00139 else
        if (tCelsius<-5)  then density := 0.00215 else
                               density := 0.00325;
      end
    else
      begin
        // here approximating with a function might speed up
        if (tCelsius<=16) then
              begin
                if (tCelsius<=-10)  then density := 0.00236 else
                if (tCelsius<= -5)  then density := 0.00332 else
                if (tCelsius<=  0)  then density := 0.00485 else
                if (tCelsius<=  2)  then density := 0.00557 else
                if (tCelsius<=  4)  then density := 0.00637 else
                if (tCelsius<=  6)  then density := 0.00727 else
                if (tCelsius<=  8)  then density := 0.00828 else
                if (tCelsius<= 10)  then density := 0.00941 else
                if (tCelsius<= 12)  then density := 0.01067 else
                if (tCelsius<= 14)  then density := 0.01208 else
                {if (tCelsius<= 16)}  {then} density :=0.01365;
              end
             else
              begin
                if (tCelsius<= 18)  then density := 0.01539 else
                if (tCelsius<= 20)  then density := 0.01732 else
                if (tCelsius<= 22)  then density := 0.01944 else
                if (tCelsius<= 24)  then density := 0.02181 else
                if (tCelsius<= 26)  then density := 0.02440 else
                if (tCelsius<= 28)  then density := 0.02726 else
                if (tCelsius<= 30)  then density := 0.03039 else
                if (tCelsius<= 35)  then density := 0.03963 else
                if (tCelsius<= 40)  then density := 0.05017 else
                if (tCelsius<= 45)  then density := 0.06545 else
                if (tCelsius<= 50)  then density := 0.08300 else
                if (tCelsius<= 60)  then density := 0.13000 else
                if (tCelsius<= 70)  then density := 0.19800 else
                if (tCelsius<= 80)  then density := 0.29300 else
                if (tCelsius<= 90)  then density := 0.42400 else
                if (tCelsius<= 95)  then density := 0.50450 else
                if (tCelsius<= 100)  then density := 0.5977 else
                if (tCelsius<= 120)  then density := 1.1220 else
                if (tCelsius<= 140)  then density := 1.9670 else
                if (tCelsius<= 170)  then density := 4.1220 else
                if (tCelsius<= 200)  then density := 7.8570 else
                if (tCelsius<= 250)  then density := 19.980 else
                if (tCelsius<= 300)  then density := 46.240 else
                if (tCelsius<= 350)  then density := 113.60 else
                                  density := 328;
              end;
      end;

   if w.isOcean[i][j] then altitude:=0 else altitude := w.elevation[i][j];
   Result := density * weightOnAltitudeProQuadrateMeter(altitude+l*TMdlConst.distance_atm_layers,i,j,w) * w.area_of_degree_squared[j];
end;

end.
