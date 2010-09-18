unit watercycle;
{(c) 2010 HB9TVM    Source code is under GPL}

interface

uses statechanges, datastructure, climaconstants, flux, Sysutils, Math;

procedure formSteam(var clima : TClima; var w : TWorld; var s : TSolarSurface; i, j : Longint; day : Longint);
procedure moveSteamUp(var clima : TClima; var w : TWorld; var s : TSolarSurface; l, i, j : Longint);
procedure moveSteam(wind : PGridShortInt; steam : PGrid; copySteam : PGrid);
procedure moveSteamDown(var clima : TClima; var w : TWorld; var s : TSolarSurface; l, i, j : Longint);
procedure decreaseRainTimes(var clima : TClima);

procedure rainSteam(var clima : TClima; var w : TWorld; i, j : Longint);

implementation

procedure formSteam(var clima : TClima; var w : TWorld; var s : TSolarSurface; i, j : Longint; day : Longint);
var
    energyToBoil,
    evaporationPct,
    evaporationQty : TClimateType;
begin
 if (clima.isIce[i][j]) {or
    (not isInSunlight(i, j, s))} then Exit;

 evaporationPct := evaporatePercentage(clima, clima.T_ocean_terr[i][j],i, j);
 if evaporationPct = 0 then Exit;

 // steam over ocean
 if w.isOcean[i][j] then
     begin
       // check how much steam we could form
       evaporationQty := (maxWaterInSteam(clima, w, 0, i, j) - clima.steam[0][i][j]) * (1/TSimConst.steam_hours)
                          * evaporationPct;
     end
  else
    // steam produced by river and lakes
    begin
        if (clima.water_surface[i][j]=0) then Exit;

        // vegetation (where it rains more than one time, slows down evaporation)
        if clima.rain_times[i][j]>0 then
          evaporationQty := clima.water_surface[i][j]/clima.rain_times[i][j]*evaporationPct  * (1/TSimConst.steam_hours)
        else
          evaporationQty := clima.water_surface[i][j]*evaporationPct * (1/TSimConst.steam_hours);

        if (day mod TSimConst.decrease_rain_times = 0) then
        begin
         Dec(clima.rain_times[i][j]);
         if clima.rain_times[i][j]<0 then clima.rain_times[i][j]:=0;
        end;

    end;


 if (evaporationQty>0) then  // here we will rain in a later procedure call
       begin
       // energy to boil the steam
       if (TPhysConst.kT_boil - clima.T_atmosphere[0][i][j]) > 0 then
          energyToBoil := TPhysConst.cp_steam * evaporationQty * (TPhysConst.kT_boil - clima.T_atmosphere[0][i][j])
         else
          energyToBoil := 0;


       if (clima.energy_ocean_terr[i][j]>=energyToBoil) then
                 // the atmosphere has enough energy to carry the steam
                 begin
                  clima.steam[0][i][j] := clima.steam[0][i][j] + evaporationQty;
                  clima.energy_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] - energyToBoil;

                  if (not w.isOcean[i][j]) then
                    clima.water_surface[i][j] := clima.water_surface[i][j] - evaporationQty;
                 end;

       end;
end;

procedure moveSteamUp(var clima : TClima; var w : TWorld; var s : TSolarSurface; l, i, j : Longint);
var
  availableSteam,
  maxSteam,
  transferEnergy,
  evaporationPct : TClimateType;
begin
 // let's compute first the temperature on the upper layer based on the thermic gradient
 clima.T_atmosphere[l][i][j] :=  thermicGradient(w, clima, l*TMdlConst.distance_atm_layers, clima.T_atmosphere[0][i][j], i, j);
                                 // we do not add altitude here, as altitude is already in clima.T_atmosphere[0][i][j]
 // if there is ice on ground or no steam to push up we exit
 if clima.isIce[i][j] or (clima.steam[l-1][i][j]=0) then Exit;
 evaporationPct := evaporatePercentage(clima, clima.T_atmosphere[l-1][i][j],i, j);
 if evaporationPct = 0 then Exit;

 // how much steam could stay in the upper layer l?
 maxSteam := maxWaterInSteam(clima, w, l, i, j);
 // how much steam is available for transfer
 availableSteam := Math.Min(clima.steam[l-1][i][j], maxSteam) * (1/TSimConst.steam_hours) * evaporationPct;
 // is there enough energy to perform the transfer to the upper layer?
 transferEnergy := availableSteam * TPhysConst.grav_acc * TMdlConst.distance_atm_layers;
 if (clima.energy_atmosphere[i][j]>transferEnergy) then
      begin
        // let's move it up
        clima.energy_atmosphere[i][j] := clima.energy_atmosphere[i][j] - transferEnergy;
        clima.steam[l-1][i][j]          := clima.steam[l-1][i][j] - availableSteam;
        clima.steam[l][i][j]        := clima.steam[l][i][j] + availableSteam;
      end;

end;


procedure moveSteamDown(var clima : TClima; var w : TWorld; var s : TSolarSurface; l, i, j : Longint);
var
  transferSteam,
  maxSteam,
  transferEnergy : TClimateType;
begin
 // recalculate temperature
 clima.T_atmosphere[l][i][j] := thermicGradient(w, clima, l*TMdlConst.distance_atm_layers, clima.T_atmosphere[0][i][j], i, j);
 if (clima.steam[l][i][j]=0) then Exit;

 // how much steam could stay in the upper layer l
 maxSteam := maxWaterInSteam(clima, w, l, i, j);
 // how much steam has to be transferred down
 transferSteam := (clima.steam[l][i][j] - maxSteam) * (1/TSimConst.steam_hours);
 if (transferSteam<0) then Exit;
 // energy which is given back to atmosphere
 transferEnergy := transferSteam * TPhysConst.grav_acc * TMdlConst.distance_atm_layers;

 // let's move it dowm
 clima.energy_atmosphere[i][j] := clima.energy_atmosphere[i][j] + transferEnergy;
 clima.steam[l][i][j]          := clima.steam[l][i][j]   - transferSteam;
 clima.steam[l-1][i][j]        := clima.steam[l-1][i][j] + transferSteam;
end;

procedure moveSteam(wind : PGridShortInt; steam : PGrid; copySteam : PGrid);
begin
 moveParticlesInAtm(wind, steam, copySteam);
end;

procedure rainSteam(var clima : TClima; var w : TWorld; i, j : Longint);
var availableSteam,
    maxWaterinAir,
    thermicEnergy   : TClimateType;
begin
 maxWaterInAir := maxWaterInSteam(clima, w, 0, i, j);

 availableSteam := (clima.steam[0][i][j] - maxWaterInAir) * (1/TSimConst.rain_hours);
 if (availableSteam<0) then
       begin
         clima.rain[i][j] := false;
       end
    else
       begin
         
         // drop the exceeding steam in rain
         if not w.isOcean[i][j] then
             clima.water_surface[i][j] := clima.water_surface[i][j] + availableSteam;
         Inc(clima.rain_times[i][j]);
         clima.rain[i][j] := true;
         clima.steam[0][i][j] :=  clima.steam[0][i][j] - availableSteam;


         // assumption: thermic energy and potential energy of clouds
         // are given back to terrain as energy of movement
         if (TPhysConst.kT_boil - clima.T_atmosphere[0][i][j])>0 then
           thermicEnergy := availableSteam * TPhysConst.cp_steam * (TPhysConst.kT_boil - clima.T_atmosphere[0][i][j])
         else
           thermicEnergy := 0;

         // give the thermic energy and potential energy to the terrain and atmosphere
         // clima.energy_atmosphere[i][j] := clima.energy_atmosphere[i][j] + potEnergy;
         clima.energy_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] +
                                          thermicEnergy;

       end;
   // compute humidity
   computeHumidity(clima, w, i, j);
end;

procedure decreaseRainTimes(var clima : TClima);
var i, j : Longint;
begin
 for j:=0 to 179 do
  for i:=0 to 359 do
    Dec(clima.rain_times[i][j]);
end;


end.
