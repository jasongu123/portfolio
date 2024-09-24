import java.util.Random;
import java.util.Scanner;

public class mmethod {
	
	static int chooseFromDist(double[] p) {
		double x=0;
		double y=0;
		for(int i=1;i<p.length+1;i++) {
			x=x+p[i-1];
			y=Math.random();
			for(int j=0;j<p.length;j++) {
			if(y<x) {
				return i;
			}
			}
		}
		return p.length;
	}
	
	static int[] rollDice(int NDice, int NSides) {
		int[] results=new int[NDice];
		Random r=new Random();
		for(int i=0;i<NDice;i++) {
			results[i]=r.nextInt(NSides)+1;
		}
		return results;
	}
	static double[] chooseDice(int[] Score, int[][][] LoseCount, int[][][] WinCount, int NDice, int M) {
		double[] diceprob=new double[NDice];
		double[] winprob=new double[NDice];
		for(int i=1;i<NDice+1;i++) {
			if(WinCount[Score[0]][Score[1]][i]+LoseCount[Score[0]][Score[1]][i]==0) {
				winprob[i-1]=0.5;
			}
			else {
			winprob[i-1]=WinCount[Score[0]][Score[1]][i]/(WinCount[Score[0]][Score[1]][i]+LoseCount[Score[0]][Score[1]][i]);
			}
		}
			double b=winprob[0];
			int bindex=0;
			for(int j=0;j<winprob.length;j++) {
				if(winprob[j]>b) {
					b=winprob[j];
					bindex=j;
				}
			}
			int rolls=0;
			for(int k=1;k<NDice+1;k++) {
				rolls=rolls+(WinCount[Score[0]][Score[1]][k]+LoseCount[Score[0]][Score[1]][k]);
			}
			diceprob[bindex]=(rolls*winprob[bindex]+M)/(rolls*winprob[bindex]+(NDice)*M);
			double g=0;
			for(int p=0;p<NDice;p++) {
				if(p!=bindex) {
					g=g+winprob[p];
				}
			}
			double v=1-diceprob[bindex];
			for(int s=0;s<NDice;s++) {
				if(s!=bindex) {
					diceprob[s]=(v*((winprob[s])*rolls+M))/(rolls*g+(NDice-1)*M);
				}
			}
			return diceprob;
		}
	static void PlayGame(int NDice,int NSides,int LTarget,int UTarget,int[][][] LoseCount,int[][][] WinCount,int M){
		int[] Scores=new int[2];
		int[] trace1=new int[LTarget+1];
		int[] trace2=new int[LTarget+1];
		int[] tr1=new int[LTarget+1];
		int[] tr2=new int[LTarget+1];
		trace1[0]=0;
		trace2[0]=0;
		int pos1=1;
		int pos2=1;
		int r1=0;
		int r2=0;
		int rtimes1=0;
		int rtimes2=0;
		while(Scores[0]<LTarget&&Scores[1]<LTarget) {
			double[] chances1=chooseDice(Scores, LoseCount, WinCount, NDice, M);
			int rolls1=chooseFromDist(chances1);
			int[] results1=rollDice(rolls1, NSides);
			double[] chances2=chooseDice(Scores, LoseCount, WinCount, NDice, M);
			int rolls2=chooseFromDist(chances2);
			int[] results2=rollDice(rolls2, NSides);
			for(int i:results1) {
				Scores[0]=Scores[0]+i;
			}
			rtimes1++;
			if(Scores[0]<LTarget) {
				trace1[pos1]=Scores[0];
				pos1++;
			}
			if(Scores[0]<LTarget) {
			for(int i:results2) {
				Scores[1]=Scores[1]+i;
			}
			rtimes2++;
			if(Scores[1]<LTarget) {
				trace2[pos2]=Scores[1];
				pos2++;
			}
			}
			tr1[r1]=results1.length;
			r1++;
			tr2[r2]=results2.length;
			r2++;
		}
			if(Scores[0]>=LTarget&&Scores[0]<=UTarget) {
				for(int i=0;i<rtimes1;i++) {
					WinCount[trace1[i]][trace2[i]][tr1[i]]++;
				}
				for(int j=0;j<rtimes2;j++){
					LoseCount[trace2[j]][trace1[j+1]][tr2[j]]++;
					}
				}
			if(Scores[0]>UTarget) {
				for(int r=0;r<rtimes1;r++) {
					LoseCount[trace1[r]][trace2[r]][tr1[r]]++;
					}
				for(int j=0;j<rtimes2;j++){
					WinCount[trace2[j]][trace1[j+1]][tr2[j]]++;
					}
				}
			if(Scores[1]>=LTarget&&Scores[1]<=UTarget) {
				for(int k=0;k<rtimes2;k++) {
					WinCount[trace2[k]][trace1[k+1]][tr2[k]]++;
				}
				for(int p=0;p<rtimes1;p++){
					LoseCount[trace1[p]][trace2[p]][tr1[p]]++;
					}
				}
			if(Scores[1]>UTarget) {
				for(int r=0;r<rtimes2;r++) {
					LoseCount[trace2[r]][trace1[r+1]][tr2[r]]++;
					}
				for(int j=0;j<rtimes1;j++){
					WinCount[trace1[j]][trace2[j]][tr1[j]]++;
					}
				}
		}
	
		static void extractAnswer(int[][][] WinCount, int[][][] LoseCount) {
			int[][] bestcoins = new int[WinCount.length][WinCount.length];
			double[][] bestprobs=new double[WinCount.length][WinCount.length];
			int bestcoin=0;
			double bestprob=0;
			double prob=0;
			int totalsuccess=0;
			int totalfailure=0;
			int wins=0;
			int losses=0;
			for(int i=0;i<WinCount.length;i++) {
				for(int j=0;j<WinCount[i].length;j++) {
					if(j==0&&i!=0) {
						bestcoins[i][j]=0;
						bestprobs[i][j]=0;
						continue;
					}
					bestprob=0;
					bestcoin=0;
					for(int k=0;k<WinCount[i][j].length-1;k++) {
						
						wins=WinCount[i][j][k+1];
						losses=LoseCount[i][j][k+1];
						prob=(float)wins/(wins+losses);
						if((!(j==0&&i!=0))&&(prob>bestprob)) {
							bestprob=prob;
							bestcoin=k+1;
						}
					}
					bestcoins[i][j]=bestcoin;
					bestprobs[i][j]=bestprob;
				}
			}
			System.out.println("Play=");
			for(int p=0;p<WinCount.length;p++){
				for(int r=0;r<WinCount.length;r++) {
					System.out.print(bestcoins[p][r]+"     ");
				}
				System.out.println("");
			}
			System.out.println("Prob=");
			for(int t=0;t<WinCount.length;t++){
				for(int y=0;y<WinCount.length;y++) {
					System.out.print(bestprobs[t][y]+"     ");
				}
				System.out.println("");
			}
		}
		
		static void prog3(int NDice, int NSides, int LTarget, int UTarget, int NGames, int M) {
			int[][][] WinCount=new int[LTarget][LTarget][NDice+1];
			int[][][] LoseCount=new int[LTarget][LTarget][NDice+1];
			for(int i=0;i<NGames;i++) {
			PlayGame(NDice,NSides,LTarget,UTarget,LoseCount,WinCount, M);
			}
			extractAnswer(WinCount, LoseCount);
		}
	
	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		System.out.print("Enter number of dice:\n");
		int NDice=Integer.parseInt(scan.next());
		System.out.print("Enter number of sides on the dice:\n");
		int NSides=Integer.parseInt(scan.next());
		System.out.print("Enter the lower target:\n");
		int LTarget=Integer.parseInt(scan.next());
		System.out.print("Enter the higher target:\n");
		int UTarget=Integer.parseInt(scan.next());
		System.out.print("Enter the number of games:\n");
		int NGames=Integer.parseInt(scan.next());
		System.out.print("Enter the hyperparameter:\n");
		int M=Integer.parseInt(scan.next());
		prog3(NDice, NSides, LTarget, UTarget, NGames, M);
	}
}
