unit nodetables;
{
   TDbNodeTable contains the client nodes seen on the GPU network
   active and inactive

   (c) by 2010 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;


type TDbNodeRow = record
    id : Longint;
    defaultservername,
    nodeid,
    nodename,
    country,
    region,
    city,
    zip,
    ip,
    port,
    localip,
    os,
    cputype,
    version          : String;
    acceptincoming   : Boolean;
    gigaflops,
    ram,
    mhz,
    nbcpus,
    bits             : Longint;
    issmp,
    isht,
    isScreensaver,
    online,
    updated          : Boolean;
    userid           : String;
    uptime,
    totaluptime      : TDateTime;
    longitude,
    latitude         : Real;
end;


type TDbNodeTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insertOrUpdate(row : TDbNodeRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbNodeTable.Create(filename : String);
begin
  inherited Create(filename, 'tbnode', 'id');
  createDbTable();
end;

procedure TDbNodeTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('defaultservername', ftInteger);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('country', ftString);
      FieldDefs.Add('region', ftString);
      FieldDefs.Add('city', ftString);
      FieldDefs.Add('zip', ftString);
      FieldDefs.Add('ip', ftString);
      FieldDefs.Add('port', ftString);
      FieldDefs.Add('localip', ftString);
      FieldDefs.Add('os', ftString);
      FieldDefs.Add('cputype', ftString);
      FieldDefs.Add('version', ftString);
      FieldDefs.Add('acceptincoming', ftBoolean);
      FieldDefs.Add('gigaflops', ftInteger);
      FieldDefs.Add('ram', ftInteger);
      FieldDefs.Add('mhz', ftInteger);
      FieldDefs.Add('bits', ftInteger);
      FieldDefs.Add('nbcpus', ftInteger);
      FieldDefs.Add('online', ftBoolean);
      FieldDefs.Add('updated', ftBoolean);
      FieldDefs.Add('uptime', ftFloat);
      FieldDefs.Add('totaluptime', ftFloat);
      FieldDefs.Add('longitude', ftFloat);
      FieldDefs.Add('latitude', ftFloat);
      FieldDefs.Add('userid', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbNodeTable.insertOrUpdate(row : TDbNodeRow);
var options : TLocateOptions;
    updated : Boolean;
begin
  options := [];
  if dataset_.Locate('nodeid', row.nodeid, options) then
    begin
      dataset_.Edit;
      updated := true;
    end
  else
    begin
      dataset_.Append;
      updated := false;
    end;

  dataset_.FieldByName('defaultservername').AsString := row.defaultservername;
  dataset_.FieldByName('nodeid').AsString := row.nodeid;
  dataset_.FieldByName('nodename').AsString := row.nodename;
  dataset_.FieldByName('country').AsString := row.country;
  dataset_.FieldByName('region').AsString := row.region;
  dataset_.FieldByName('city').AsString := row.city;
  dataset_.FieldByName('zip').AsString := row.zip;
  dataset_.FieldByName('ip').AsString := row.ip;
  dataset_.FieldByName('port').AsString := row.port;
  dataset_.FieldByName('localip').AsString := row.localip;
  dataset_.FieldByName('os').AsString := row.os;
  dataset_.FieldByName('cputype').AsString := row.cputype;
  dataset_.FieldByName('version').AsString := row.version;
  dataset_.FieldByName('acceptincoming').AsBoolean := row.acceptincoming;
  dataset_.FieldByName('gigaflops').AsInteger := row.gigaflops;
  dataset_.FieldByName('ram').AsInteger := row.ram;
  dataset_.FieldByName('mhz').AsInteger := row.mhz;
  dataset_.FieldByName('nbcpus').AsInteger := row.nbcpus;
  dataset_.FieldByName('bits').AsInteger := row.bits;
  dataset_.FieldByName('online').AsBoolean := row.online;
  dataset_.FieldByName('updated').AsBoolean := true;
  dataset_.FieldByName('uptime').AsFloat := row.uptime;
  dataset_.FieldByName('totaluptime').AsFloat := row.totaluptime;
  dataset_.FieldByName('latitude').AsFloat := row.latitude;
  dataset_.FieldByName('longitude').AsFloat := row.longitude;
  dataset_.FieldByName('userid').AsString := row.userid;

  if updated then
    dataset_.FieldByName('update_dt').AsDateTime := Now
  else
    begin
      dataset_.FieldByName('create_dt').AsDateTime := Now;
      //dataset_.FieldByName('update_dt').AsDateTime := nil;
    end;


  dataset_.Post;
  dataset_.ApplyUpdates;
end;


end.
