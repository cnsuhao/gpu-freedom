unit jobmanagementforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, sqlite3conn, sqldb, db, BufDataset, memds, dbf, FileUtil,
  LResources, Forms, Controls, Graphics, Dialogs, ExtCtrls, ComCtrls, StdCtrls,
  Spin, DBGrids, DbCtrls, jobapis, coreobjects;

type

  { TJobManagementForm }

  TJobManagementForm = class(TForm)
    btnSelectInputWorkunit: TButton;
    btnSubmitJob: TButton;
    cbJobBox: TComboBox;
    cbJobType: TComboBox;
    cbRequireAck: TCheckBox;
    cbTagOutputWorkunit: TCheckBox;
    cgWorkflow: TCheckGroup;
    cbTagInputWorkunit: TCheckBox;
    datasource: TDatasource;
    dbgJobQueue: TDBGrid;
    DBNavigator: TDBNavigator;
    edtWorkunitInput: TEdit;
    edtWorkunitOutput: TEdit;
    gbWorkunits: TGroupBox;
    lblNumberOfRequests: TLabel;
    lblOutputWorkunitName: TLabel;
    lblWorkunitInput: TLabel;
    lblJobType: TLabel;
    lblJob: TLabel;
    lblJobDefinitionIdDesc: TLabel;
    lblJobDefinitionId: TLabel;
    OpenDialog: TOpenDialog;
    pnCreateJob: TPanel;
    rbGlobal: TRadioButton;
    rbLocal: TRadioButton;
    seNbRequests: TSpinEdit;
    SQLite3conn: TSQLite3Connection;
    SQLQuery: TSQLQuery;
    SQLTransaction: TSQLTransaction;
    procedure btnSubmitJobClick(Sender: TObject);
    procedure datasourceDataChange(Sender: TObject; Field: TField);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure SQLite3connAfterConnect(Sender: TObject);

  private

  public

  end;

var
  JobManagementForm: TJobManagementForm;

implementation

{ TJobManagementForm }

procedure TJobManagementForm.FormCreate(Sender: TObject);
begin
  Visible := True;
  sqlite3Conn.DatabaseName := 'gpucore.db';
  sqlite3Conn.Connected := true;

  SQLTransaction.Database:=SQLite3Conn;

  SQLQuery.Database:=SQLite3Conn;
  SQLQuery.SQL.text:='select * from tbjobqueue';
  SQLQuery.open;

  DataSource.DataSet:=SQLQuery;

  dbgJobQueue.DataSource:=DataSource;
  dbgJobQueue.AutoFillColumns:=true;

  dbNavigator.DataSource := DataSource;

end;

procedure TJobManagementForm.btnSubmitJobClick(Sender: TObject);
var job : TGPUJobApi;
begin
  job.job        := cbJobBox.Text;
  job.jobtype    := 'GPU_Engine';
  job.islocal    := rbLocal.Checked;
  job.requireack := cbRequireAck.Checked;
  job.trandetails.nbrequests     := seNbRequests.Value;
  job.trandetails.workunitjob    := Trim(edtWorkunitInput.Text);
  job.trandetails.tagwujob       := cbTagInputWorkunit.Checked;
  job.trandetails.workunitresult := Trim(edtWorkunitOutput.Text);
  job.trandetails.tagwuresult    := cbTagOutputWorkunit.Checked;

  jobapi.createJob(job);
  lblJobDefinitionId.Caption := job.jobdefinitionid;

end;

procedure TJobManagementForm.datasourceDataChange(Sender: TObject; Field: TField);
begin

end;

procedure TJobManagementForm.FormDestroy(Sender: TObject);
begin
  //
end;

procedure TJobManagementForm.SQLite3connAfterConnect(Sender: TObject);
begin

end;


initialization
  {$I jobmanagementforms.lrs}

end.

