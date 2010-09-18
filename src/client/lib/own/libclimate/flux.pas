unit flux;
// this unit computes wind and currents
{(c) 2010 HB9TVM    Source code is under GPL}

interface

uses datastructure, Math, climaconstants, SysUtils, conversion, averages;

const
    
    WIND  = 0;
    SURFACE_AND_MARINE_CURRENT  = 1;

procedure moveEnergy(var clima : TClima; energy_grid : PGrid; copy_energy_grid : PGrid; temp_grid : PGrid; direction : PGridShortInt;
                     typeFlux : Integer; var w : TWorld; transfEnergy : Boolean);
procedure applyCoriolis(source_flow_grid, target_flow_grid : PGridShortInt; apply : Boolean);
procedure followSurface(var w : TWorld; flow_grid : PGridShortInt);

procedure moveParticlesInAtm(wind : PGridShortInt; particles : PGrid; copyParticles : PGrid);

implementation

procedure moveEnergy(var clima : TClima; energy_grid : PGrid; copy_energy_grid : PGrid; temp_grid : PGrid; direction : PGridShortInt;
                     typeFlux : Integer; var w : TWorld; transfEnergy : Boolean);
var i, j : Longint;
    check_north,
    check_south,
    check_west,
    check_east : Longint;

    T_north,
    T_south,
    T_west,
    T_east,
    T_north_west,
    T_north_east,
    T_south_west,
    T_south_east,
    T_own,
    T_lowestCardinal,
    T_lowestDiagonal,
    T_lowest : TClimatetype;

    e_flux : TClimateType;

    lat : TClimateType;

  function transferEnergy(i_source, j_source, i_target, j_target : Longint; directionFlow : Byte) : Boolean;
  var
    energy_transferred,
    finalEnergy,
    T_radiated,
    contact_height,
    elevation,
    lat,
    factor      : TClimateType;
  begin
     Result := False;

     direction^[i][j] := directionFlow;
     // used for upper atmospheric layers
     if (not transfEnergy) then
              begin
                 Result := true;
                 Exit;
              end;

     if copy_energy_grid^[i_source][j_source]<0 then raise Exception.Create('Energy negative at i:'+IntToStr(i)+' and j:'+IntToStr(j));

     if (typeFlux=WIND) then
          begin
             // this is an additional factor to avoid melting of points
             factor := w.length_of_degree[j_source]/w.length_of_degree[LatToY(0)];
             if factor>1 then factor := 1;
             T_radiated := clima.T_atmosphere[0][i][j];
             energy_transferred := Abs(energy_grid^[i_source][j_source]-energy_grid^[i_target][j_target])
                                   * e_flux * TSimConst.pct_wind_transfer * factor;

             {
             // using Stefan Boltzmann the atmosphere is too static
             elevation := w.elevation[i][j];
             if (elevation<0) then elevation := 0;
             contact_height := TSimConst.max_atmosphere_height - elevation;
             energy_transferred := TPhysConst.stefan_boltzmann * Power(T_radiated,4)
                           * Avg(w.length_of_degree[j], w.length_of_degree[j_target])
                           * contact_height * TSimConst.hour_step * e_flux *
                           TSimConst.pct_wind_transfer;
             }
             if energy_transferred<0 then raise Exception.create('Energy transferred between squares is negative');
          end
        else
          begin
           T_radiated := clima.T_ocean_terr[i][j];
           contact_height := Avg(Abs(w.elevation[i_target][j_target]), Abs(w.elevation[i][j]));

           energy_transferred := TPhysConst.stefan_boltzmann * Power(T_radiated,4)
                           * Avg(w.length_of_degree[j], w.length_of_degree[j_target]) * contact_height * TSimConst.hour_step * e_flux;
           if energy_transferred<0 then raise Exception.create('Energy transferred between squares is negative');
          end;


     finalEnergy := energy_grid^[i_source][j_source] - energy_transferred;
     if (finalEnergy<0) then Exit;

     energy_grid^[i_source][j_source] := finalEnergy;
     energy_grid^[i_target][j_target] := energy_grid^[i_target][j_target] + energy_transferred;

     Result := True;
  end;

begin
// we need a local copy of the energy grid
for j:=0 to 179 do
 for i:=0 to 359 do
    copy_energy_grid^[i][j] := energy_grid^[i][j];


for j:=0 to 179 do
 for i:=0 to 359 do
    begin

      // initialize with the correct flux factor
      if (typeFlux=WIND) then
             e_flux := (1/TSimConst.exchange_flux_atm)
      else
      if (typeFlux=SURFACE_AND_MARINE_CURRENT) then
         begin
            if w.isOcean[i][j] and (not clima.isIce[i][j]) then
                       e_flux := (1/TSimConst.exchange_flux_ocean)
            else
               e_flux := (1/TSimConst.exchange_flux_terrain);
         end
      else
           raise Exception.Create('typeFlux is unknown');

      check_north := j-1;
      check_south := j+1;
      check_west  := i-1;
      check_east  := i+1;

      // we live on a sphere
      if check_north<0 then check_north := 179;
      if check_south>179 then check_south := 0;
      if check_west<0 then check_west := 359;
      if check_east>359 then check_east := 0;

      T_north := temp_grid^[i][check_north];
      T_south := temp_grid^[i][check_south];
      T_west  := temp_grid^[check_west][j];
      T_east  := temp_grid^[check_east][j];
      T_north_west := temp_grid^[check_west][check_north];
      T_north_east := temp_grid^[check_east][check_north];
      T_south_west := temp_grid^[check_west][check_south];
      T_south_east := temp_grid^[check_east][check_south];
      T_own   := temp_grid^[i][j];


      T_lowestCardinal  := Math.Min(Math.min(T_north, T_south), Math.Min(t_west, T_east));
      T_lowestDiagonal := Math.Min(Math.min(T_north_east, T_south_west), Math.Min(t_north_west, T_south_east));
      T_lowest := Math.Min(T_lowestCardinal, T_lowestDiagonal);

      if T_lowest > T_own then continue;

      // we move the energy into the place with lower temperature
      // energy transfer between ocean and terrain is possible
      // NOTE: the direction of winds and currents is depending on invert_flow flag

      if (T_own = T_lowest) then
         begin
           direction^[i][j] := NONE;
         end
      else
      if (T_north = T_lowest) then
         begin
              transferEnergy(i , j, i, check_north, SOUTH*TSimConst.invert_flow);
         end else
      if (T_south = T_lowest) then
         begin
              transferEnergy(i , j, i, check_south, NORTH*TSimConst.invert_flow);
         end else
      if (T_west = T_lowest) then
         begin
              transferEnergy(i , j, check_west, j, EAST*TSimConst.invert_flow);
         end else
      if (T_east = T_lowest) then
         begin
              transferEnergy(i , j, check_east, j, WEST*TSimConst.invert_flow);
         end else
      if (T_south_west = T_lowest) then
         begin
              transferEnergy(i , j, check_west, check_south, NORTH_EAST*TSimConst.invert_flow);
         end else
      if (T_south_east = T_lowest) then
         begin
              transferEnergy(i , j, check_east, check_south, NORTH_WEST*TSimConst.invert_flow);
         end else
      if (T_north_west = T_lowest) then
         begin
              transferEnergy(i , j, check_west, check_north, SOUTH_EAST*TSimConst.invert_flow);
         end else
      if (T_north_east = T_lowest) then
         begin
              transferEnergy(i , j, check_east, check_north, SOUTH_WEST*TSimConst.invert_flow);
         end;
    end;
end;

procedure applyCoriolis(source_flow_grid, target_flow_grid : PGridShortInt; apply : Boolean);
var i, j : Longint;

procedure coriolisClockwise(i,j : Longint);
begin
         // clockwise turn
         case source_flow_grid^[i][j] of
            NORTH : target_flow_grid^[i][j] := NORTH_WEST;
            SOUTH : target_flow_grid^[i][j] := SOUTH_EAST;
            EAST  : target_flow_grid^[i][j] := NORTH_EAST;//EAST;
            WEST  : target_flow_grid^[i][j] := SOUTH_WEST;//WEST;
            NORTH_EAST : target_flow_grid^[i][j] := NORTH;
            NORTH_WEST : target_flow_grid^[i][j] := WEST;
            SOUTH_EAST : target_flow_grid^[i][j] := EAST;
            SOUTH_WEST : target_flow_grid^[i][j] := SOUTH;
         end;
end;

procedure coriolisCounterClockwise(i,j : Longint);
begin
        // counterclockwise turn
         case source_flow_grid^[i][j] of
            NORTH : target_flow_grid^[i][j] := NORTH_WEST;
            SOUTH : target_flow_grid^[i][j] := SOUTH_EAST;
            EAST  : target_flow_grid^[i][j] := NORTH_EAST;//EAST;
            WEST  : target_flow_grid^[i][j] := SOUTH_WEST;//WEST;
            NORTH_EAST : target_flow_grid^[i][j] := NORTH;
            NORTH_WEST : target_flow_grid^[i][j] := WEST;
            SOUTH_EAST : target_flow_grid^[i][j] := EAST;
            SOUTH_WEST : target_flow_grid^[i][j] := SOUTH;
         end;
end;

begin
if apply then
 begin
  // north half sphere
  for j:=0 to 89 do
    for i:=0 to 359 do
      if (TMdlConst.inverse_rotation=1) then
        coriolisClockwise(i, j) // normal rotation
      else
        coriolisCounterClockwise(i, j);

   // south half sphere
   for j:=90 to 179 do
     for i:=0 to 359 do
      if (TMdlConst.inverse_rotation=1) then
        coriolisCounterClockwise(i, j) // normal rotation
      else
        coriolisClockwise(i, j);

 end
else
  begin
     // only copying if there is no coriolis
     for j:=0 to 179 do
      for i:=0 to 359 do
          target_flow_grid^[i][j] := source_flow_grid^[i][j];
  end;
end;

procedure followSurface(var w : TWorld; flow_grid : PGridShortInt);
var
 i, j, m_north, m_south, m_west, m_east : Longint;
 E_north,
 E_south,
 E_west,
 E_east,
 E_north_west,
 E_north_east,
 E_south_west,
 E_south_east,
 E_own,
 E_lowestCardinal,
 E_lowestDiagonal,
 E_lowest : TClimatetype;


    function directionClose(direction1, direction2 : ShortInt) : Boolean;
    begin
      Result := false;
      if (direction1=direction2) then
                     begin
                       Result := true;
                       Exit;
                     end;

      case direction1 of
         NORTH : Result := (direction2=NORTH_WEST) or (direction2=NORTH_EAST);
         SOUTH : Result := (direction2=SOUTH_WEST) or (direction2=SOUTH_EAST);
         EAST  : Result := (direction2=NORTH_EAST) or (direction2=SOUTH_EAST);
         WEST  : Result := (direction2=NORTH_WEST) or (direction2=SOUTH_WEST);
         NORTH_EAST : Result := (direction2=NORTH) or (direction2=EAST);
         NORTH_WEST : Result := (direction2=NORTH) or (direction2=WEST);
         SOUTH_EAST : Result := (direction2=SOUTH) or (direction2=EAST);
         SOUTH_WEST : Result := (direction2=SOUTH) or (direction2=WEST);
      end;

    end;


    procedure changeDir(i, j : Longint; proposedDir : ShortInt);
    begin
      if directionClose(flow_grid^[i][j], proposedDir) then
          flow_grid^[i][j] := proposedDir;
    end;

begin
for j:=0 to 179 do
 for i:=0 to 359 do
  begin
   if w.isOcean[i][j] then continue;

   m_north := j-1;
   m_south := j+1;
   m_west  := i-1;
   m_east  := i+1;

   // we live on a sphere
   if m_north<0 then m_north := 179;
   if m_south>179 then m_south := 0;
   if m_west<0 then m_west := 359;
   if m_east>359 then m_east := 0;

   // check the surface gradient
   E_north := w.elevation[i][m_north];
   E_south := w.elevation[i][m_south];
   E_west  := w.elevation[m_west][j];
   E_east  := w.elevation[m_east][j];
   E_north_west := w.elevation[m_west][m_north];
   E_north_east := w.elevation[m_east][m_north];
   E_south_west := w.elevation[m_west][m_south];
   E_south_east := w.elevation[m_east][m_south];
   E_own   := w.elevation[i][j];

   E_lowestCardinal  := Math.Min(Math.min(E_north, E_south), Math.Min(E_west, E_east));
   E_lowestDiagonal := Math.Min(Math.min(E_north_east, E_south_west), Math.Min(E_north_west, E_south_east));
   E_lowest := Math.Min(E_lowestCardinal, E_lowestDiagonal);


   if (E_own = E_lowest) then continue
     else
      if (E_north = E_lowest) then
         begin
              changeDir(i , j, NORTH);
         end else
      if (E_south = E_lowest) then
         begin
              changeDir(i , j, SOUTH);
         end else
      if (E_west = E_lowest) then
         begin
              changeDir(i , j, WEST);
         end else
      if (E_east = E_lowest) then
         begin
              changeDir(i , j, EAST);
         end else
      if (E_south_west = E_lowest) then
         begin
              changeDir(i , j, SOUTH_WEST);
         end else
      if (E_south_east = E_lowest) then
         begin
              changeDir(i , j, SOUTH_EAST);
         end else
      if (E_north_west = E_lowest) then
         begin
              changeDir(i , j, NORTH_WEST);
         end else
      if (E_north_east = E_lowest) then
         begin
              changeDir(i , j, NORTH_EAST);
         end;
 end;


end;

procedure moveParticlesInAtm(wind : PGridShortInt; particles : PGrid; copyParticles : PGrid);
var
    direction : Shortint;
    i, j,
    target_i,
    target_j : Longint;

    speed,
    particlesQuantityCenter,
    particlesQuantityCardinal,
    particlesQuantityDiagonal : TClimateType;

    m_north,
    m_south,
    m_west,
    m_east   : Longint;

    procedure diffuseParticles(direction : ShortInt; particlesQty : TClimateType);
    begin
        if (direction = NONE) then
              particles^[i][j] := particles^[i][j] + particlesQty
        else
        if (direction = NORTH) then
              particles^[i][m_north] := particles^[i][m_north] + particlesQty
        else
        if (direction = SOUTH) then
              particles^[i][m_south] := particles^[i][m_south] + particlesQty
        else
        if (direction = WEST) then
           particles^[m_west][j] := particles^[m_west][j] + particlesQty
        else
        if (direction = EAST) then
           particles^[m_east][j] := particles^[m_east][j] + particlesQty
        else
        if (direction = NORTH_EAST) then
           particles^[m_east][m_north] := particles^[m_east][m_north] + particlesQty
        else
        if (direction = NORTH_WEST) then
           particles^[m_west][m_north] := particles^[m_west][m_north] + particlesQty
        else
        if (direction = SOUTH_EAST) then
           particles^[m_east][m_south] := particles^[m_east][m_south] + particlesQty
        else
        if (direction = SOUTH_WEST) then
           particles^[m_west][m_south] := particles^[m_west][m_south] + particlesQty;
    end;

begin
// we need copy of the particles grid and we clear the actual one
for j:=0 to 179 do
 for i:=0 to 359 do
   begin
    copyParticles^[i][j] := particles^[i][j];
    particles^[i][j] := 0;
   end;


for j:=0 to 179 do
 for i:=0 to 359 do
  begin
   m_north := j-1;
   m_south := j+1;
   m_west  := i-1;
   m_east  := i+1;

   // we live on a sphere
   if m_north<0 then m_north := 179;
   if m_south>179 then m_south := 0;
   if m_west<0 then m_west := 359;
   if m_east>359 then m_east := 0;

   // TODO: adjust speed, and quantities for better diffusion effect
   // if step=1h, degree_step=15
   // speed := (TSimConst.degree_step/15);

   // now we simply move the particles according to the wind
   // diffusion percentages computed by
   // D:\Projects\gpu_solar\src\dllbuilding\thedayaftertomorrow\docs\flux-computation.png
   particlesQuantityCenter := copyParticles^[i][j]*0.1414710605;
   particlesQuantityCardinal := copyParticles^[i][j]*0.1374730640;
   ParticlesQuantityDiagonal := copyParticles^[i][j]*0.07715917088;

   diffuseParticles(wind^[i][j], particlesQuantityCenter);
   diffuseParticles(wind^[i][m_north], particlesQuantityCardinal);
   diffuseParticles(wind^[i][m_south], particlesQuantityCardinal);
   diffuseParticles(wind^[m_east][j], particlesQuantityCardinal);
   diffuseParticles(wind^[m_west][j], particlesQuantityCardinal);
   diffuseParticles(wind^[m_east][m_north], particlesQuantityDiagonal);
   diffuseParticles(wind^[m_east][m_south], particlesQuantityDiagonal);
   diffuseParticles(wind^[m_west][m_north], particlesQuantityDiagonal);
   diffuseParticles(wind^[m_west][m_south], particlesQuantityDiagonal);

  end; // for
end;


end.
