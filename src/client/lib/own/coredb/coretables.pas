unit coretables;
{
   TDbCoreTable is the ancenstor of all tables in the core database
   The tables are stored in a file in sqlite format.

   (c) by 2010 HB9TVM and the GPU Team
}
interface

uses sqlite3ds;

type TDbCoreTable = class(TObject)
  public
    constructor Create(filename, tablename, primarykey : String);

    function getDS() : TSqlite3Dataset;

    procedure Open();
    procedure Close();

    procedure execSQL(sql : String);

  protected
    dataset_ : TSqlite3Dataset;
end;

implementation


constructor TDbCoreTable.Create(filename, tablename, primarykey : String);
begin
  inherited Create;

  dataset_ := TSqlite3Dataset.Create(nil);
  dataset_.filename   := filename;
  dataset_.tablename  := tablename;
  dataset_.PrimaryKey := primarykey;
end;

function TDbCoreTable.getDS() : TSqlite3Dataset;
begin
 Result := dataset_;
end;

procedure TDbCoreTable.Open();
begin
 dataset_.Open;
end;


procedure TDbCoreTable.Close();
begin
 dataset_.Close;
end;

procedure TDbCoreTable.execSQL(sql : String);
begin
 dataset_.execSQL(sql);
 dataset_.RefetchData;
end;

end.
