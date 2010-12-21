unit retrievedtables;
{
   TDbRetrievedTable contains the latest message received from each server
    for each channel or jobqueue.

   (c) by 2010 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbRetrievedRow = record
    id,
    lastmsg,
    server_id    : Longint;
    msgtype      : String;
    create_dt,
    update_dt    : TDateTime;
end;

type TDbRetrievedTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure getRow(row : TDbRetrievedRow; server_id : Longint;
                     channame, chantype : String);
    procedure insertOrUpdate(row : TDbRetrievedRow);

  private
    procedure createDbTable();
    function  createMsgType(server_id : Longint;
                            channame, chantype : String) : String;
  end;

implementation


constructor TDbRetrievedTable.Create(filename : String);
begin
  inherited Create(filename, 'tbretrieved', 'id');
  createDbTable();
end;

procedure TDbRetrievedTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('lastmsg', ftInteger);
      FieldDefs.Add('msgtype', ftString);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;


function TDbRetrievedTable.createMsgType(server_id : Longint;
                                         channame, chantype : String) : String;
begin
  Result := chantype+'_'+channame+'_'+IntToStr(server_id);
end;


procedure TDbRetrievedTable.getRow(row : TDbRetrievedRow; server_id : Longint;
                                   channame, chantype : String);
var options : TLocateOptions;
    msgtype : String;
begin
 msgtype := createMsgType(server_id, channame, chantype);
 options := [];
 if dataset_.Locate('msgtype', msgtype, options) then
   begin
     row.id      := dataset_.FieldByName('id').AsInteger;
     row.lastmsg := dataset_.FieldByName('lastmsg').AsInteger;
     row.create_dt := dataset_.FieldByName('create_dt').AsDateTime;
   end
  else
   begin
     // we construct a new row
     row.id := -1;
     row.lastmsg := -1;
     row.create_dt := Now;
   end;

   row.server_id := server_id;
   row.msgtype := msgtype;
end;

procedure TDbRetrievedTable.insertOrUpdate(row : TDbRetrievedRow);
var options : TLocateOptions;
    updated : Boolean;
begin
  options := [];
  if dataset_.Locate('id', row.id, options) then
    begin
      dataset_.Edit;
      updated := true;
    end
  else
    begin
      dataset_.Append;
      updated := false;
    end;

  dataset_.FieldByName('lastmsg').AsInteger := row.lastmsg;
  dataset_.FieldByName('msgtype').AsString  := row.msgtype;
  dataset_.FieldByName('create_dt').AsDateTime := row.create_dt;

  if updated then
    dataset_.FieldByName('update_dt').AsDateTime := Now;

  dataset_.Post;
  dataset_.ApplyUpdates;
end;

end.
