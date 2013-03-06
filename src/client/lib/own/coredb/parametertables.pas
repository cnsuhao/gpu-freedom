unit parametertables;
{
   TDbParameterTable contains parameters used by core and frontend.

   (c) by 2010-2013 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils, identities;


type TDbParameterRow = record
    id            : Longint;
    paramtype,
    paramname,
    paramvalue    : String;
    create_dt,
    update_dt   : TDateTime;
end;

type TDbParameterTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insert(row : TDbParameterRow);

    function retrieveParameter(paramtype, paramname, ifmissing : String) : String;
    function setParameter(paramtype, paramname, paramvalue : String) : Boolean;

  private
    procedure createDbTable();
  end;

implementation

constructor TDbParameterTable.Create(filename : String);
begin
  inherited Create(filename, 'tbparameter', 'id');
  createDbTable();
end;


procedure TDbParameterTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('paramtype', ftString);
      FieldDefs.Add('paramname', ftString);
      FieldDefs.Add('paramvalue', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbParameterTable.insert(row : TDbParameterRow);
var options : TLocateOptions;
    updated : Boolean;
begin
  // TODO remove this from public visibility

  dataset_.Append;

  dataset_.FieldByName('paramtype').AsString := row.paramtype;
  dataset_.FieldByName('paramname').AsString := row.paramname;
  dataset_.FieldByName('paramvalue').AsString := row.paramvalue;
  dataset_.FieldByName('create_dt').AsDateTime := row.create_dt;
  dataset_.FieldByName('update_dt').AsDateTime := row.update_dt;

  dataset_.Post;
  dataset_.ApplyUpdates;
end;

{
// TODO: delete me
// returns the new id
function TDbChannelTable.retrieveLatestChat(channame, chantype : String; lastid : Longint;
                                            var content : AnsiString; ownChat : Boolean) : Longint;
var newid : Longint;
    sqlStatement : String;
begin
  newid := lastid;
  dataset_.Close();
  sqlStatement := 'select * from tbchannel where (externalid>'+IntToStr(lastid)+') '+
                  'and (channame='+QUOTE+channame+QUOTE+') '+
                  'and (chantype='+QUOTE+chantype+QUOTE+') ';
  if not ownChat then sqlStatement := sqlStatement +
                  'and (nodeid<>'+QUOTE+myGPUID.nodeid+QUOTE+') ';
  sqlStatement := sqlStatement +
                  ' order by externalid asc;';

  dataset_.SQL := sqlStatement;
  dataset_.Open;

  content := '';
  dataset_.First;
  while not dataset_.EOF do
     begin
       content := content +
                  dataset_.FieldByName('nodename').AsString+'>'+
                  dataset_.FieldByName('content').AsString+#13#10;
       newid := dataset_.FieldByName('externalid').AsLongint;
       dataset_.Next;
     end;

  Result := newid;
end;
}


function TDbParameterTable.retrieveParameter(paramtype, paramname, ifmissing : String) : String;
begin
 WriteLn('TODO: implement retrieveParameter');
end;

function TDbParameterTable.setParameter(paramtype, paramname, paramvalue : String) : Boolean;
begin
 WriteLn('TODO: implement setParameter');
end;

end.
