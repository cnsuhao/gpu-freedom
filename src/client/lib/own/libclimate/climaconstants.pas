unit climaconstants;
{(c) 2010 HB9TVM    Source code is under GPL}
{$mode objfpc}
{$static on}

interface
  uses
    datastructure;

    type
      TMdlConst = class

      albedo,               {natural reflection of earth}
      tau_visible,          {how much visible light is absorbed by atmosphere}
      tau_infrared,         {how much infrared light is absorbed by atmosphere}
      distanceFromSun       {distance from Sun in Astronomic Units}
                      :    TClimateType; static;

      atmospheric_layers,   {number of atmospheric layers}
      distance_atm_layers,  {in m}
      initDegreeSunlight,   // where the sun initially shines
      inverse_rotation
                     : Longint; static;

      rotation,            // if the earth rotates around its axis
      revolution           // if the earth rotates around the sun
                         : Boolean; static;


      class procedure Init;
    end;

    type
      TInitCond = class
        thermic_poles,          // lowest temperature at poles}
        thermic_excursion,      // temp_equator = thermic_poles+thermic_excursion}
        surface_shift,          // additional thermic shift for surface
        ocean_shift,            // shift applied to ocean
        terrain_shift,          // shift applied to terrain

        desert_belt_lat,        // latitude of desert belt
        desert_belt_ext,        // desert_belt_latitude +- desert_belt_extension in angle degrees
        desert_belt_delta_T,    // degrees Kelvin added to the belt
                                // and subtracted to the equator
        thermic_gradient_avg,   // thermic gradient that reduces temperature for each 100 on surface m
        thermic_gradient_sea    // how much degree celsius the temperature is decreased per 1000 m sea depth
                              : TClimateType; static;
      class procedure Init;
    end;

 type
   TSimConst = class

    riverandlakes_pct,        // how much water flows down in river, not trapped by a lake
    cloud_reflection_pct,     // humidity times this factor gives a reflection of solar energy
    cloud_isolation_pct,      // isolation of clouds stop radiation of thermic energy into space
    co2_isolation_pct,        // isolation due to CO2 stop radiation of thermic energy into space -> greenhouse effect
    rain_hours,               // how many hours it takes to rain down the >100% humidity
    steam_hours,              // how many hours it takes to form steam
    radiation_hours,          // how many hours are needed to radiate energy back into space
    exchange_atm_terr,        // how many hours it takes to exchange energy between terrain and atmosphere
    exchange_flux_ocean,      // how many hours it takes to exchange energy between ocean squares
    exchange_flux_atm,        // how many hours it takes to exchange energy between atmosphere squares
    exchange_flux_terrain,    // how many hours it takes to transfer energy between terrain squares
    deltaTterrestrialEnergy,  // how many degree Kelvin per hour the earth warms terrain up
    pct_wind_transfer,        // how much energy is transferred per hour
    paint_river_pct,          // when to paint a river in the simulation, meaning how much percent of water it
                              // is above average.
    paint_clouds,             // when to paint clouds {kg/m^2}

    thermic_gradient_dry,
    thermic_gradient_wet   :  TClimateType; static;

    invert_flow,              // set to -1 to invert wind flux

    hour_step,  {s}
    degree_step, {deg}
    decrease_rain_times,      // after how many days a rain unit is removed

    max_atmosphere_height,    {in m}
    earth_crust_height,       {in m}
    max_terrain_height       {in m}
                            : Longint;  static;

    evaporation_start_temp,                          // where evaporation starts (in Celsius)
    full_evaporation_temp  : TClimateType; static;  // where evaporation is full (in Celsius)

    // CO2
    population,
    energy_source_oil : Boolean; static;
    population_increase_pct,
    co2_production_per_human_per_year,
    co2_production_vulcano,
    co2_absorption_vegetation,
    co2_absorption_ocean  : TClimateType; static;

    class procedure Init;
  end;

  type TPhysConst = class
    cp_water,         {J/(kg K)}
    cp_steam,         {J/(kg K)}
    cp_air,           {J/(kg K) at 20 degree C}
    cp_earth,         {J/(kg K)}
    cp_ice,           {J/(kg K)}

    length_of_degree_square,   {m, a degree is about 111 km}
    area_of_degree_square_world_flat,   //111320 * 111320 {m^2}

    density_earth,  {kg /m^3}
    density_water,  {kg /m^3}
    density_ice,    {kg /m^3}

    absolutezero,
    kT_ice,
    kT_boil,

    earth_inclination_on_ecliptic, {in degree}

    stefan_boltzmann,  {W/(m^2 K^4)}

    SolarConstant, {W/m^2}

    grav_acc,          {in m/s^2}
    pressure_on_surface_at_sea_level, {N / m^2}
    weight_on_surface_at_sea_level,  {kg / m^2}

    earth_radius {in m}
                   :  TClimateType; static;

    class procedure Init;
  end;

type TSpecialParam = class
    // vulcan
    vulcan : Boolean; static;

    vulcan_lat,
    vulcan_lon     : Longint; static;
    vulcan_ashes_pct,
    vulcan_hours : TClimateType; static;

    // thermonuclear bomb
    nuclear_bomb,
    nuclear_war  : Boolean; static;

    nuclear_bomb_lat,
    nuclear_bomb_lon  : Longint; static;

    nuclear_bomb_energy,
    nuclear_ashes_pct,
    nuclear_war_hours,
    ashes_fallout_pct         : TClimateType; static;
    ashes_cycle_active        : Boolean; static;

    class procedure Init;
end;


procedure initModelParameters;


implementation

class procedure TMdlConst.Init;
begin
    albedo := 0;         {natural reflection of earth}
    tau_visible := 0.8;  {how much visible light is absorbed by atmosphere}
    tau_infrared := 0.1;  {how much infrared light is absorbed by atmosphere}
    distanceFromSun := 1; {distance from Sun in Astronomic Units}

    atmospheric_layers := 5; {number of atmospheric layers}
    distance_atm_layers := 1500 {in m};

    rotation := true;
    revolution := true;
    inverse_rotation := 1;
    initDegreeSunlight := 0;

end;

class procedure TInitCond.Init;
begin
    // initial conditions
    thermic_poles := -21.3;       {lowest temperature at poles (+thermic_shift)}
    thermic_excursion := 63;      {temp_equator = thermic_poles+thermic_excursion+thermic_shift}
    surface_shift := 0;           // additional thermic shift for surface
    ocean_shift   := 0.2;         // shift applied to ocean
    terrain_shift := 0.8;         // shift applied to terrain

    desert_belt_lat     := 10;        // latitude of desert belt
    desert_belt_ext     := 25;         // desert_belt_latitude +- desert_belt_extension in angle degrees
    desert_belt_delta_T := 0;         // degrees Kelvin added to the belt
                                     // and subtracted to the equator
    thermic_gradient_avg := 0.6667;   // thermic gradient that reduces temperature for each 100 on surface m
    thermic_gradient_sea := 0.9;  // how much degree celsius the temperature is decreased per 1000 m sea depth
end;

class procedure TSimConst.Init;
begin
    riverandlakes_pct := 0.55; // how much water flows down in river, not trapped by a lake
    decrease_rain_times := 3; // after how many days a rain unit is removed

    cloud_reflection_pct  := 0.44; // humidity times this factor gives a reflection of solar energy
    cloud_isolation_pct := 0.4; // isolation of clouds stop radiation of thermic energy into space
    co2_isolation_pct := 0.26; // greenhouse effect, value from http://en.wikipedia.org/wiki/Greenhouse_effect

    rain_hours := 1;   // how many hours it takes to rain down the >100% humidity
    steam_hours := 1;  // how many hours it takes to form steam
    radiation_hours := 1; // how many hours are needed to radiate energy back into space
    exchange_atm_terr := 72; // how many hours it takes to exchange energy between terrain and atmosphere
    exchange_flux_ocean := 48; // how many hours it takes to exchange energy between ocean squares
    exchange_flux_atm := 1; // how many hours it takes to exchange energy between atmosphere squares
    exchange_flux_terrain := 192; // how many hours it takes to transfer energy between terrain squares
    deltaTterrestrialEnergy := 0; // how many degree Kelvin per hour the earth warms terrain up

    thermic_gradient_dry := 1;
    thermic_gradient_wet := 0.6;

    invert_flow := 1;  // set to -1 to invert wind flux
    pct_wind_transfer := 0.05; // how much energy is transferred per hour

    hour_step := 1200; {s}
    degree_step := 5; {deg}

    max_atmosphere_height := 14000; {in m}
    earth_crust_height := 4000; {in m}
    max_terrain_height := 9000; {in m}

    paint_river_pct := 1.2; // when to paint a river in the simulation, meaning how much percent of water it
                           // is above average
    paint_clouds  := 40; // when to paint clouds  {kg/m^2}

    // CO2
    // data estimated from http://en.wikipedia.org/wiki/Carbon_dioxide
    population := false;
    energy_source_oil := true;
    population_increase_pct := 0.01; {percent per year}
    co2_production_per_human_per_year := 27E9/6E9; {ton_co2/year  overall tons/population in 2010}
    co2_production_vulcano := 200E6; {ton_co2/year}
    co2_absorption_vegetation := 27.2E9*0.25; {ton_co2/year}
    co2_absorption_ocean := 27.2E9*0.5; {ton_co2/year}

    evaporation_start_temp := 0;    // where evaporation starts
    full_evaporation_temp := 100;    // where evaporation is 100%
end;

class procedure TPhysConst.Init;
begin
    cp_water := 4182; {J/(kg K)}
    cp_steam := 1863; {J/(kg K)}
    cp_air   := 1005; {J/(kg K) at 20 degree C}
    cp_earth := 790; {J/(kg K)}
    cp_ice   := 37E-6; {J/(kg K)}

    length_of_degree_square := 111320; {m, a degree is about 111 km}
    area_of_degree_square_world_flat := 12392142400;        //111320 * 111320 {m^2}

    density_earth := 5515; {kg /m^3}
    density_water := 1000; {kg /m^3}
    density_ice   := 917; {kg /m^3}

    absolutezero := 273.16;
    kT_ice  := 273.16;
    kT_boil := 373.16;

    earth_inclination_on_ecliptic := (23+27/60); {in degree}

    stefan_boltzmann := 5.67E-8; {W/(m^2 K^4)}

    SolarConstant := 1370; {W/m^2}

    grav_acc         := 9.80665; {in m/s^2}
    pressure_on_surface_at_sea_level := 1E5; {N / m^2}
    weight_on_surface_at_sea_level := 1E5/9.81; {kg / m^2}

    earth_radius := 6.371E6; {in m}

end;

class procedure TSpecialParam.Init;
begin
    // vulcan
    vulcan := false;

    vulcan_lat    :=  64;  // Eyjafjallajkull in Island
    vulcan_lon    :=  -17;
    vulcan_hours  := 365*24;
    vulcan_ashes_pct := 1;

    // thermonuclear bomb
    nuclear_bomb := false;

    nuclear_bomb_energy := 900;
    nuclear_bomb_lat  := 42;     // New York
    nuclear_bomb_lon  := -74;
    nuclear_ashes_pct := 1;

    nuclear_war_hours := 72;

    ashes_fallout_pct := 0.01;

    ashes_cycle_active := false;
end;


procedure initModelParameters;
begin
  TMdlConst.Init;
  TInitCond.Init;
  TSimConst.Init;
  TPhysConst.Init;
  TSpecialParam.Init;
end;

end.
