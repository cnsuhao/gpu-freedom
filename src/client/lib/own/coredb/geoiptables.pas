unit geoiptables;
{
   TDbGeoIPTable contains geoip information used by core and frontend.

   (c) by 2010-2016 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils, identities;


type TDbGeoIPRow = record
    id            : Longint;
    longitude,
    latitude      : Real;

    city,
    countryname,
    countrycode,
    timezone,
    ip,
    error : String;

    create_dt     : TDateTime;
end;

type TDbGeoIPTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insert(row : TDbGeoIPRow);

    function getLongitude() : Double;
    function getLatitude() : Double;

  private
    procedure createDbTable();
  end;

implementation

constructor TDbGeoIPTable.Create(filename : String);
begin
  inherited Create(filename, 'tbgeoip', 'id');
  createDbTable();
end;


procedure TDbGeoIPTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('longitude', ftFloat);
      FieldDefs.Add('latitude', ftFloat);
      FieldDefs.Add('city', ftString);
      FieldDefs.Add('countryname', ftString);
      FieldDefs.Add('countrycode', ftString);
      FieldDefs.Add('timezone', ftString);
      FieldDefs.Add('ip', ftString);
      FieldDefs.Add('error', ftString);

      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbGeoIPTable.insert(row : TDbGeoIPRow);
var options : TLocateOptions;
begin
   options := [];
   dataset_.Append;

   dataset_.FieldByName('longitude').AsFloat := row.longitude;
   dataset_.FieldByName('latitude').AsFloat := row.latitude;
   dataset_.FieldByName('city').AsString := row.city;
   dataset_.FieldByName('countryname').AsString := row.countryname;
   dataset_.FieldByName('countrycode').AsString := row.countrycode;
   dataset_.FieldByName('timezone').AsString := row.timezone;
   dataset_.FieldByName('ip').AsString := row.ip;
   dataset_.FieldByName('error').AsString := row.error;
   dataset_.FieldByName('create_dt').AsDateTime := Now;

   dataset_.Post;
   dataset_.ApplyUpdates;
end;


function TDbGeoIPTable.getLongitude : Double;
begin
   // TODO: implement me
   Result := 0;
end;

function TDbGeoIPTable.getLatitude : Double;
begin
  // TODO: implement me
  Result := 0;
end;


end.
