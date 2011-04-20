unit channeltables;
{
   TDbChannelTable contains data in channels used by sensors, whiteboard and other
    GPU services. Chat is also implemented as channel.

   (c) by 2010-2011 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils, identities;


type TDbChannelRow = record
    id,
    externalid    : Longint;
    content,
    user,
    nodename,
    nodeid,
    channame,
    chantype      : String;
    server_id     : Longint;
    create_dt,
    usertime_dt   : TDateTime;
end;

type TDbChannelTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insert(row : TDbChannelRow);
    // returns the new id
    function retrieveLatest(channame, chantype : String; lastid : Longint;
             var content : AnsiString) : Longint;

  private
    procedure createDbTable();
  end;

implementation

constructor TDbChannelTable.Create(filename : String);
begin
  inherited Create(filename, 'tbchannel', 'id');
  createDbTable();
end;


procedure TDbChannelTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('externalid', ftInteger);
      FieldDefs.Add('content', ftString);
      FieldDefs.Add('user', ftString);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('channame', ftString);
      FieldDefs.Add('chantype', ftString);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('usertime_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbChannelTable.insert(row : TDbChannelRow);
var options : TLocateOptions;
    updated : Boolean;
begin
  dataset_.Append;

  dataset_.FieldByName('externalid').AsInteger := row.externalid;
  dataset_.FieldByName('content').AsString := row.content;
  dataset_.FieldByName('user').AsString := row.user;
  dataset_.FieldByName('nodename').AsString := row.nodename;
  dataset_.FieldByName('nodeid').AsString := row.nodeid;
  dataset_.FieldByName('channame').AsString := row.channame;
  dataset_.FieldByName('chantype').AsString := row.chantype;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.FieldByName('create_dt').AsDateTime := row.create_dt;
  dataset_.FieldByName('usertime_dt').AsDateTime := row.usertime_dt;

  dataset_.Post;
  dataset_.ApplyUpdates;
end;

// returns the new id
function TDbChannelTable.retrieveLatest(channame, chantype : String; lastid : Longint;
                                        var content : AnsiString) : Longint;
var newid : Longint;
begin
  dataset_.Close();
  dataset_.SQL := 'select * from tbchannel where (externalid>'+IntToStr(lastid)+') '+
                  'and (channame='+QUOTE+channame+QUOTE+') '+
                  'and (chantype='+QUOTE+chantype+QUOTE+') '+
                  'and (nodeid<>'+QUOTE+myGPUID.nodeid+QUOTE+')'+
                  ' order by externalid asc;';
  dataset_.Open;

  content := '';
  dataset_.First;
  while not dataset_.EOF do
     begin
       content := content + dataset_.FieldByName('content').AsString+#13#10;
       newid := dataset_.FieldByName('externalid').AsLongint;
       dataset_.Next;
     end;

  Result := newid;
end;

end.
