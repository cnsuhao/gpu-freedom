unit datastructure;
{(c) 2010 HB9TVM    Source code is under GPL}

interface

uses SysUtils;

const
  MAX_ATM_LAYERS = 10;     {maximum number of atmospheric layers }

  ATMOSPHERE       = 100;
  OCEAN_TERR       = 200;
  AVERAGE          = 300;
  OCEAN            = 400;
  TERRAIN          = 500;
  AIR_OVER_OCEAN   = 600;
  AIR_OVER_TERRAIN = 700;

  NONE  = 0;
  NORTH = 1;
  SOUTH = -1;
  WEST  = -2;
  EAST  = 2;
  NORTH_WEST = -3;
  NORTH_EAST = 4;
  SOUTH_WEST = -4;
  SOUTH_EAST = 3;

type  TColor = Longint;
type  TClimateType = Extended;
type  TLatitude = Array[0..179] of TClimateType;
type  TGrid = Array [0..359] of Array [0..179] of TClimateType;
type  TLayersGrid = Array [0..MAX_ATM_LAYERS-1] of TGrid;
type  TGridBoolean = Array [0..359] of Array [0..179] of Boolean;
type  TGridShortInt = Array [0..359] of Array [0..179] of Shortint;
type  TLayersGridShortInt = Array [0..MAX_ATM_LAYERS-1] of TGridShortInt;
type  TGridLongint = Array [0..359] of Array [0..179] of Longint;
type  TGridColor = Array [0..359] of Array [0..179] of TColor;
type  PGrid = ^TGrid;
type  PGridShortInt = ^TGridShortInt;
type  PGridColor = ^TGridColor;

  type TTime = record
     year : Longint;
     day : Longint;
     hour : TClimateType;
  end;
  type PTime = ^TTime;

  type TSolarSurface = record
     degstart : Longint;
     degend : Longint;
  end;
  type PSolarSurface = ^TSolarSurface;

  type TWorld = record
     elevation : TGrid;
     isOcean : TGridBoolean;
     area_of_degree_squared,
     length_of_degree         : TLatitude;
  end;
  type PWorld = ^TWorld;

  type TClima = record
     energy_atmosphere : TGrid;
     energy_ocean_terr : TGrid;

     T_atmosphere : TLayersGrid; //(in Kelvin)!
     T_ocean_terr : TGrid; //(in Kelvin)!

     wind            :  TLayersGridShortInt;
     surfaceTransfer : TGridShortInt;

     steam      : TLayersGrid;
     humidity   : TGrid;

     rain          : TGridBoolean;
     water_surface : TGrid;
     rain_times    : TGridLongint;

     avgWaterSurface : TLatitude;

     population,
     co2_tons,
     ashes_pct     : TGrid;

     isIce   : TGridBoolean;
  end;
  type PClima = ^TClima;


implementation


end.
