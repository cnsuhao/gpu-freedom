unit energyfunctions;
{(c) 2010 HB9TVM    Source code is under GPL}

interface
uses climaconstants, conversion, datastructure, statechanges, Math, SysUtils;

  function computeEarthInclination(day : Longint) : TClimateType;
  function computeEnergyFactorWithAngle(i, j : Longint; earthInclination : TClimateType):TClimateType;
  function computeEnergyFromSunOnSquare(i, j : Longint; earthInclination : TClimateType; var clima : TClima; var w : TWorld) : TClimateType;
  procedure spreadEnergyOnAtmosphereAndTerrain(var clima : TClima; energy : TClimateType; i, j : Longint);
  procedure updateTemperature(var clima : TClima; var w : TWorld; i, j : Longint);
  procedure radiateTerrestrialEnergy(var clima : TClima; var w : TWorld; i, j : Longint);
  procedure exchangeEnergyBetweenAtmAndTerrain(var clima : TClima; var w : TWorld; i, j : Longint);

  function computeRadiatedEnergyIntoSpace(var clima : TClima; var w : TWorld; i, j : Longint) : TClimateType;
  procedure radiateEnergyIntoSpace(var clima : TClima; var w : TWorld; i, j : Longint);

  procedure updateIncomingEnergyOnCellGrid(var clima : TClima; var w : TWorld; var sSurface : TSolarSurface; earthInclination : TClimateType; i, j : Longint);
  procedure updateOutgoingEnergyOnCellGrid(var clima : TClima; var w : TWorld; var sSurface : TSolarSurface; earthInclination : TClimateType; i, j : Longint);


implementation


function computeEarthInclination(day : Longint) : TClimateType;
begin
  if (day<1) or (day>365) then
     raise Exception.Create('Day has to be between 1 and 365 but was '+IntToStr(day));
  {+10.5 is to reach the maximum inclination at the 22 of June, the summer solstice}
  if TMdlConst.revolution then
      Result := -sin(day/365*2*Pi+Pi/2+10.5/365*2*Pi) * TPhysConst.earth_inclination_on_ecliptic
    else
      Result := TPhysConst.earth_inclination_on_ecliptic;
end;


function computeEnergyFactorWithAngle(i, j : Longint; earthInclination : TClimateType):TClimateType;
var angle, factor : TClimateType;
begin
  angle := Abs(YtoLat(j)-earthInclination);
  factor := sin((90-angle)/90*Pi/2);
  if (factor<0) then factor := 0;
  Result := factor;
end;

function computeEnergyFromSunOnSquare(i, j : Longint; earthInclination : TClimateType; var clima : TClima; var w : TWorld): TClimateType;
var reflection,
    angle : TClimateType;
begin
  reflection := TMdlConst.albedo + clima.humidity[i][j] * TSimConst.cloud_reflection_pct + clima.ashes_pct[i][j];
  if (reflection >1) then reflection := 1;
  Result := TPhysConst.SolarConstant * (1-reflection) *
            computeEnergyFactorWithAngle(i,j, earthInclination)
            / Power(TMdlConst.distanceFromSun,2)
            * TSimConst.hour_step
            * w.area_of_degree_squared[j];

end;

procedure spreadEnergyOnAtmosphereAndTerrain(var clima : TClima; energy : TClimateType; i,j : Longint);
begin
     clima.energy_atmosphere[i][j] := energy * (TMdlConst.tau_visible) + clima.energy_atmosphere[i][j];
     clima.energy_ocean_terr[i][j] := energy * (1-TMdlConst.tau_visible) + clima.energy_ocean_terr[i][j];
end;

procedure updateTemperature(var clima : TClima; var w : TWorld; i, j : Longint);
var
  divAtmosphere,
  divOcean,
  divTerrain,
  weight : TClimateType;
begin
          if w.isOcean[i][j] then
              weight := TPhysConst.weight_on_surface_at_sea_level
          else
              weight := weightOnAltitudeProQuadrateMeter(w.elevation[i][j], i, j, w);

          divAtmosphere := (TPhysConst.cp_air * weight * w.area_of_degree_squared[j]);
          if divAtmosphere<>0 then
           clima.T_atmosphere[0][i][j] := clima.energy_atmosphere[i][j] / divAtmosphere
          else raise Exception.create('divAtmosphere is zero!');

          if w.isOcean[i][j] then
             begin
              divOcean := (TPhysConst.cp_water * Abs(w.elevation[i][j]) * w.area_of_degree_squared[j] * TPhysConst.density_water) +
                          (TPhysConst.cp_earth * (w.elevation[i][j]+TSimConst.earth_crust_height) * w.area_of_degree_squared[j] * TPhysConst.density_earth);
              if divOcean<>0 then
                clima.T_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] / divOcean
              else clima.T_ocean_terr[i][j]:= 0;
             end
             else
             begin
              // terrain
              divTerrain := TPhysConst.cp_earth * (w.elevation[i][j]+TSimConst.earth_crust_height) * w.area_of_degree_squared[j] * TPhysConst.density_earth;
              if divTerrain<>0 then
                clima.T_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] / divTerrain
              else raise Exception.create('divTerrain is zero!');

            end;
end;


procedure radiateTerrestrialEnergy(var clima : TClima; var w : TWorld; i, j : Longint);
var
  multiple_earth,
  multiple_ocean : TClimateType;
begin
  // earth constantly radiates gravitational energy to the terrain
  clima.T_ocean_terr[i][j] := clima.T_ocean_terr[i][j] + TSimConst.deltaTterrestrialEnergy*(1-TMdlConst.tau_infrared)*(TSimConst.degree_step/15);

  // earth part
  multiple_earth := (TPhysConst.cp_earth * (w.elevation[i][j]+TSimConst.earth_crust_height) * w.area_of_degree_squared[j] * TPhysConst.density_earth);

  if w.isOcean[i][j] then
              multiple_ocean := (TPhysConst.cp_water * Abs(w.elevation[i][j]) * w.area_of_degree_squared[j] * TPhysConst.density_water)
  else
              multiple_ocean := 0;

  clima.energy_ocean_terr[i][j] := clima.T_ocean_terr[i][j]*(multiple_earth+multiple_ocean);
end;

procedure exchangeEnergyBetweenAtmAndTerrain(var clima : TClima; var w : TWorld; i, j : Longint);
var energy_moved,
    finalEnergy : TClimateType;
begin
 if clima.T_ocean_terr[i][j]>clima.T_atmosphere[0][i][j] then
     begin
        // radiate energy from terrain to atmosphere
        energy_moved := TPhysConst.stefan_boltzmann * Power(clima.T_ocean_terr[i][j],4)
                        * w.area_of_degree_squared[j] * TSimConst.hour_step * (1/TSimConst.exchange_atm_terr);
        if energy_moved<0 then raise Exception.create('Energy radiated from terrain to atmosphere is negative');
        finalEnergy := clima.energy_ocean_terr[i][j] - energy_moved;
        if finalEnergy<0 then Exit;

        clima.energy_ocean_terr[i][j] := finalEnergy;
        clima.energy_atmosphere[i][j] := clima.energy_atmosphere[i][j] + energy_moved;
     end
   else
     begin
        // radiate energy from atmosphere to terrain
        energy_moved := TPhysConst.stefan_boltzmann * Power(clima.T_atmosphere[0][i][j],4)
        * w.area_of_degree_squared[j] * TSimConst.hour_step * (1/TSimConst.exchange_atm_terr);
        if energy_moved<0 then raise Exception.create('Energy radiated from atmosphere to terrain is negative');

        finalEnergy := clima.energy_atmosphere[i][j] - energy_moved;
        if finalEnergy<0 then Exit;

        clima.energy_atmosphere[i][j] := finalEnergy;
        clima.energy_ocean_terr[i][j] := clima.energy_ocean_terr[i][j] + energy_moved;
     end;
end;


function computeRadiatedEnergyIntoSpace(var clima : TClima; var w : TWorld; i, j : Longint) : TClimateType;
var
   isolation,
   co2_isolation    : TClimateType;
begin
 co2_isolation := clima.co2_tons[i][j]/1E7; // where it is red on plot it is 1
 if (co2_isolation>1) then co2_isolation := 1;

 isolation := clima.humidity[i][j] * TSimConst.cloud_isolation_pct + TSimConst.co2_isolation_pct*co2_isolation;
 if (isolation>1) then isolation := 1;
 Result := (1-isolation) * TPhysConst.stefan_boltzmann * Power(clima.T_atmosphere[0][i][j],4)
                    * w.area_of_degree_squared[j] * TSimConst.hour_step * (1/TSimConst.radiation_hours);
end;

procedure radiateEnergyIntoSpace(var clima : TClima; var w : TWorld; i, j : Longint);
var
    energy_radiated,
    finalEnergy : TClimateType;

begin
 energy_radiated := computeRadiatedEnergyIntoSpace(clima, w, i, j);
 if (energy_radiated<0) then raise Exception.create('Energy radiated into space is negative');
 finalEnergy := clima.energy_atmosphere[i][j] - energy_radiated;
 if (finalEnergy<0) then Exit;
 clima.energy_atmosphere[i][j] := finalEnergy;
end;


procedure updateIncomingEnergyOnCellGrid(var clima : TClima; var w : TWorld; var sSurface : TSolarSurface; earthInclination : TClimateType; i, j : Longint);
var
    energyIn : TClimateType;
begin
  if isInSunlight(i, j, sSurface) then
    begin
      energyIn := computeEnergyFromSunOnSquare(i, j, earthInclination, clima, w);
      spreadEnergyOnAtmosphereAndTerrain(clima, energyIn, i, j);
      updateTemperature(clima, w, i, j);
    end;

  radiateTerrestrialEnergy(clima, w, i, j);
  updateTemperature(clima, w, i, j);
end;

procedure updateOutgoingEnergyOnCellGrid(var clima : TClima; var w : TWorld; var sSurface : TSolarSurface; earthInclination : TClimateType; i, j : Longint);
begin
  updateTemperature(clima, w, i, j);
  exchangeEnergyBetweenAtmAndTerrain(clima, w, i, j);
  updateTemperature(clima, w, i, j);
  radiateEnergyIntoSpace(clima, w, i, j);
  updateTemperature(clima, w, i, j);
end;


end.
