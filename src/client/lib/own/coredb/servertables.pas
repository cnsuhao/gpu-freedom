unit servertables;
{
   TDbServerTable contains the server nodes seen on the GPU network
   active and inactive

   (c) by 2010 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbServerRow = record
    id : Longint;
    externalid       : Longint;
    servername,
    serverurl,
    chatchannel,
    version          : String;
    online,
    updated,
    defaultsrv,
    superserver      : Boolean;
    uptime,
    totaluptime      : TDateTime;
    longitude,
    latitude,
    distance         : Real;
    activenodes,
    jobsinqueue       : Longint;
end;


type TDbServerTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insertOrUpdate(row : TDbServerRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbServerTable.Create(filename : String);
begin
  inherited Create(filename, 'tbserver', 'id');
  createDbTable();
end;

procedure TDbServerTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('externalid', ftInteger);
      FieldDefs.Add('servername', ftString);
      FieldDefs.Add('serverurl', ftString);
      FieldDefs.Add('chatchannel', ftString);
      FieldDefs.Add('version', ftString);
      FieldDefs.Add('online', ftBoolean);
      FieldDefs.Add('updated', ftBoolean);
      FieldDefs.Add('defaultsrv', ftBoolean);
      FieldDefs.Add('superserver', ftBoolean);
      FieldDefs.Add('uptime', ftFloat);
      FieldDefs.Add('totaluptime', ftFloat);
      FieldDefs.Add('longitude', ftFloat);
      FieldDefs.Add('latitude', ftFloat);
      FieldDefs.Add('distance', ftFloat);
      FieldDefs.Add('activenodes', ftInteger);
      FieldDefs.Add('jobinqueue', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;


procedure TDbServerTable.insertOrUpdate(row : TDbServerRow);
var options : TLocateOptions;
    updated : Boolean;
begin
  options := [];
  if dataset_.Locate('externalid', row.externalid, options) then
    begin
      dataset_.Edit;
      updated := true;
    end
  else
    begin
      dataset_.Append;
      updated := false;
    end;

  dataset_.FieldByName('externalid').AsInteger := row.externalid;
  dataset_.FieldByName('servername').AsString := row.servername;
  dataset_.FieldByName('serverurl').AsString := row.serverurl;
  dataset_.FieldByName('chatchannel').AsString := row.chatchannel;
  dataset_.FieldByName('version').AsString := row.version;
  dataset_.FieldByName('online').AsBoolean := row.online;
  dataset_.FieldByName('updated').AsBoolean := true;
  dataset_.FieldByName('defaultsrv').AsBoolean := row.defaultsrv;
  dataset_.FieldByName('superserver').AsBoolean := row.superserver;
  dataset_.FieldByName('uptime').AsFloat := row.uptime;
  dataset_.FieldByName('totaluptime').AsFloat := row.totaluptime;
  dataset_.FieldByName('latitude').AsFloat := row.latitude;
  dataset_.FieldByName('longitude').AsFloat := row.longitude;
  dataset_.FieldByName('distance').AsFloat := row.distance;
  dataset_.FieldByName('activenodes').AsInteger := row.activenodes;
  dataset_.FieldByName('jobinqueue').AsInteger := row.jobsinqueue;

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
