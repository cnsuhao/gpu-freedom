unit earthtextures;

interface

uses SysUtils, Math, texturestructure, utils, colors, conversions;

const
 POP_INCREASE_PCT = 0.01;

type TEarthTexturer = class(TSphereTexturer)
  public
   constructor Create;
   function getTexture() : PGridColor; virtual;
private
   appPath_    : String;
   isOcean_    : TGridBoolean;
   elevation_,
   population_ : TGridFloat;
   colors_     : TGridColor;

   procedure loadElevation;
   procedure loadPopulation;
   procedure increasePopulation(days : Longint);
   procedure paintTexture;
end;

implementation

constructor TEarthTexturer.Create;
begin
 appPath_ := ExtractFilePath(ParamStr(0));
 loadElevation;
 loadPopulation;
 increasePopulation(20*365); // the dataset is of 1990, simulation begins in 2010
 paintTexture;
end;

function TEarthTexturer.getTexture() : PGridColor;
begin
  Result := @colors_;
end;

procedure TEarthTexturer.loadElevation;
var G    : TextFile;
    i, j : Longint;
    str  : String;
begin
 // load elevations
 AssignFile(G, appPath_+'data\planet\planet-elevation.txt');
 try
  Reset(G);
  for j := 0 to T_HEIGHT  do
   begin
      for i:=0 to T_WIDTH do
       begin
             ReadLn(G, str);
             if (Trim(str)<>'') then
               begin
                elevation_[i][j] := StrToFloat(str);
                isOcean_[i][j] := elevation_[i][j] <=0;
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
    str        : String;
    x,y        : Longint;
begin
// load population
 for j := 0 to T_HEIGHT  do
    for i:=0 to T_WIDTH do
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
                 population_[x][y]:=population_[x][y]+population;
               end;
       end;
  finally
   Closefile(G);
  end;

end;

procedure TEarthTexturer.increasePopulation(days : Longint);
var base, exp, factor : TFloatType;
    i,j    : Longint;
begin
  base := 1;
  exp  := 1 + (POP_INCREASE_PCT*days/365);
  factor := Power(base, exp);
  for j:=0 to T_HEIGHT do
    for i:=0 to T_WIDTH do
       population_[i][j] := population_[i][j]*factor;
end;

procedure TEarthTexturer.paintTexture();
var i, j : Longint;
begin
  for j:=0 to T_HEIGHT do
    for i:=0 to T_WIDTH do
      begin
         if isOcean_[i][j] then
             colors_[i][j] := clLightBlue
            else
             colors_[i][j] := clBrown;
      end;
end;

end.
