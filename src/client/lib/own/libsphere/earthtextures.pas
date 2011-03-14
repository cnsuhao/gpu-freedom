unit earthtextures;

interface

uses texturestructure;

const
 POP_INCREASE_PCT = 0.01;

type TEarthTexturer = class(TSphereTexturer);
  public
   constructor Create;
   procedure fillTexture(var colors : TGridColor); virtual; override;
private
   appPath_    : String;
   isOcean_    : TGridBoolean;
   elevation_,
   population_ : TGridFloat;

   procedure loadElevation;
   procedure loadPopulation;
   procedure increasePopulation(days : Longint);
end;

implementation

constructor TEarthTexturer.Create;
begin
 appPath_ := ExtractFilePath(ParamStr(0));
end;

procedure TEarthTexturer.fillTexture(var colors : TGridColor); virtual; override;
begin
  //TODO: fill me
end;

procedure TEarthTexturer.loadElevation;
var G    : TextFile;
    i, j : Longint;
begin
 // load elevations
 AssignFile(G, appPath_+'data\planet\planet-elevation.txt');
 try
  Reset(G);
  for j := 0 to 179  do
   begin
      for i:=0 to 359 do
       begin
             ReadLn(G, str);
             if (Trim(str)<>'') then
               begin
                elevation_[i][j] := StrToFloat(str);
                isOcean_[i][j] := w.elevation[i][j] <=0;
               end
              else
               raise Exception.Create('Problem in file planet-elevation.txt');
       end;
   end;
  finally
   Closefile(G);
  end;
end;

procedure TEarthTexturer.loadPopulation;
var G          : TextFile;
    i, j,
    population : Longint;
begin
// load population
 for j := 0 to 179  do
    for i:=0 to 359 do
       population_[i][j]:=0;
try
  AssignFile(G, appPath_+'data\planet\planet-population-1990.txt');
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
                 population_[x][y]:=clima.population[x][y]+population;
               end;
       end;
  finally
   Closefile(G);
  end;
  increasePopulation(2*3650); // the dataset is of 1990, simulation begins in 2010
end;

procedure TEarthTexturer.increasePopulation(days : Longint);
var base, exp, factor : TClimateType;
    i,j    : Longint;
begin
  base := 1;
  exp  := 1 + (POP_INCREASE_PCT*days/365);
  factor := Power(base, exp);
  for j:=0 to 179 do
    for i:=0 to 359 do
       population_[i][j] := population[i][j]_*factor;
end;

end.
