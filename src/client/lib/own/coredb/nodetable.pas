unit nodetable;
{
   TDbNodeTable contains the client nodes seen on the GPU network
   active and inactive

   (c) by 2010 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, coretable;


type TDbNodeRow = record
    id : Longint;
    defaultserver_id : Longint;
    nodeid,
    nodename,
    country,
    region,
    ip               : String
    port             : Longint;
    localip,
    os,
    cputype,
    version          : String;
    acceptincoming   : Boolean;
    gigaflops,
    ram,
    mhz,
    nbcpus           : Longint;
    online,
    updated          : Boolean;
    uptime,
    totaluptime      : TDateTime;
    longitude,
    latitude         : Real;
end;


type TDbNodeTable = class(TDbCoreTable)
  public
    constructor Create(filename);

    function insertOrUpdate(row : TDbNodeRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbNodeTable.Create(filename);
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
      FieldDefs.Add('defaultserver_id', ftInteger);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('country', ftString);
      FieldDefs.Add('region', ftString);
      FieldDefs.Add('ip', ftString);
      FieldDefs.Add('port', ftInteger);
      FieldDefs.Add('localip', ftString);
      FieldDefs.Add('os', ftString);
      FieldDefs.Add('cputype', ftString);
      FieldDefs.Add('version', ftString);
      FieldDefs.Add('acceptincoming', ftBoolean);
      FieldDefs.Add('gigaflops', ftInteger);
      FieldDefs.Add('ram', ftInteger);
      FieldDefs.Add('mhz', ftInteger);
      FieldDefs.Add('nbcpus', ftInteger);
      FieldDefs.Add('online', ftBoolean);
      FieldDefs.Add('updated', ftBoolean);
      FieldDefs.Add('uptime', ftFloat);
      FieldDefs.Add('totaluptime', ftFloat);
      FieldDefs.Add('longitude', ftFloat);
      FieldDefs.Add('latitude', ftFloat);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

function TDbNodeTable.insertOrUpdate(row : TDbNodeRow);
var options : TLocateOptions;
    updated : Boolean;
begin
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

  dataset_.FieldByName('defaultserver_id').AsInteger := row.defaultserver_id;
  dataset_.FieldByName('nodeid').AsString := row.nodeid;
  dataset_.FieldByName('nodename').AsString := row.nodename;
  dataset_.FieldByName('country').AsString := row.country;
  dataset_.FieldByName('region').AsString := row.region;
  dataset_.FieldByName('ip').AsString := row.ip;
  dataset_.FieldByName('port').AsInteger := row.port;
  dataset_.FieldByName('localip').AsString := row.localip;
  dataset_.FieldByName('os').AsString := row.os;
  dataset_.FieldByName('cputype').AsString := row.cputype;
  dataset_.FieldByName('version').AsString := row.version;
  dataset_.FieldByName('acceptincoming').AsBoolean := row.acceptincoming;
  dataset_.FieldByName('gigaflops').AsInteger := row.gigaflops;
  dataset_.FieldByName('ram').AsInteger := row.ram;
  dataset_.FieldByName('mhz').AsInteger := row.mhz;
  dataset_.FieldByName('nbcpus').AsInteger := row.nbcpus;
  dataset_.FieldByName('online').AsBoolean := row.online;
  dataset_.FieldByName('updated').AsBoolean := true;
  dataset_.FieldByName('uptime').AsFloat := row.uptime;
  dataset_.FieldByName('totaluptime').AsFloat := row.totaluptime;
  dataset_.FieldByName('latitude').AsDateTime := row.latitude;
  dataset_.FieldByName('longitude').AsDateTime := row.longitude;

  if updated then
    dataset_.FieldByName('update_dt').AsDateTime := Now();
  else
    begin
      dataset_.FieldByName('create_dt').AsDateTime := Now();
      dataset_.FieldByName('update_dt').AsDateTime := nil;
    end;


  dataset_.Post;
  datset_.ApplyUpdates;
end;


end.
