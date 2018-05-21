import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import com.alibaba.fastjson.JSON;

public class Judge {
	private Map<String, String> predictionMapUser = new HashMap<String, String>();
	private Map<String, String> predictionMapSku = new HashMap<String, String>();
    /**
     * F1-Score 评分计算
     * 注意：尽量减少时间和空间开销，catch住一切异常
     * @param standardResultFile 标准结果文件: id, ret, A/B
     * @param userCommitFile 用户提交结果: id, ret
     * @param evalStrategy 评分类型: 0 表示A榜评分， 1表示B榜评分
     * @return 评测结果数组，第0位表示错误类型err-type, 第1位为错误码err-code，第2位为总分score, 第3位表示子得分个数n, 之后n位分别表示F1,F2...,Fn.
     *         err-type: 0.0f表示结果正常, 1.0f 表示编码错误（文件类型错误、字符集错误或者BOM问题）; 2.0f 表示文件格式错误（行数错误、字段数错误、字段分隔符或换行符错误等）; 3.0f表示文件大小错误（文件太小或太大）; 4.0f 表示逻辑错误（如含有重复结果等）;5.0f 表示提交的是老的数据.
     *         err-code 当err-type不为0时，err-code会直接返回给参赛者
     *
     */
    public float[] judge(String standardResultFile, String userCommitFile, int evalStrategy) {
		float[] result = new float[6];
		result[0] = 0;
		result[1] = 0;
		result[3] = 2;
		int tp = 0, fp = 0, fn = 0, tp1 = 0, fp1 = 0, fn1 = 0;
		String separator = ",";
		String separatorUser = ","; // 推荐使用统一的标准csv格式，用英文半角","进行csv文件分割
		try {
			// 如果csv文件内容比较复杂，如长文本，则请使用标准csv库，或者自行考虑处理文本中分隔符、换行符等问题。
			if(new File(userCommitFile).length()>2048000){
				result[0]=3;
				return result;
			}
			BufferedReader brUser = new BufferedReader(
					new InputStreamReader(new FileInputStream(userCommitFile), "utf-8"));
			BufferedReader brStand = new BufferedReader(
					new InputStreamReader(new FileInputStream(standardResultFile), "utf-8"));
			String lineUser;
			String lineStand;
			int i = 0;
			while ((lineUser = brUser.readLine()) != null) {
				if (0 == i) {
					i++;
					continue;
				}
				if (lineUser.isEmpty()) {
					continue;
				}
				String[] parts = lineUser.split(separatorUser);
				
				try{
					Integer.parseInt(parts[0]);
					Integer.parseInt(parts[1]);
				} catch(NumberFormatException e){
					System.out.println("decode error !!!It isn't Integer");
					result[0]=1;
					return result;
				}
				
				if (parts.length != 2) {
					System.out.println("userCommitFile error !!!");
					result[0]=2;
					return result;
				}
				
				//判断是否老数据
				if (Integer.parseInt(parts[0]) < 200000) {
					result[0]=5;
					return result;
				}
				
				if (predictionMapUser.get(parts[0]) != null) {
					System.out.println("duplicate records !!!");
					result[0]=4;
					return result;
				}
				
				predictionMapUser.put(parts[0], parts[1]);
				if (predictionMapSku.get(parts[0]) != null) {
					System.out.println("duplicate records !!!");
					result[0]=4;
					return result;
				}
				i++;
			}

			// 如果要求必须测试集所有条目均预测，则对测试条目进行检查，否则，可跳过
			// if (i != 200001) {
			// System.out.println("userCommitFile less than 20W rows, error
			// !!!");
			// return -1.0f;
			// }
			brUser.close();
			int usersize = i;
			i = 0;
			int sum_abs=0
			int sum_abs_square=0
			int sum_count=0
			while ((lineStand = brStand.readLine()) != null) {
				if (0 == i) {
					i++;
					continue;
				}
				String[] parts = lineStand.split(separator);
				// if(!predictionMap.containsKey(parts[0])) {
				// System.out.println("user file not cantain record: " +
				// parts[0]);
				// return -1.0f;
				// }
				Integer.parseInt(parts[0]);
				Integer.parseInt(parts[1]);
				predictionMapSku.put(parts[0],parts[1])
				if ((evalStrategy == 0 && parts[2].equals("A")) || (evalStrategy == 1 && parts[2].equals("B"))) { // 仅计算A榜或B榜
					String predictionLabeluser = predictionMapUser.get(parts[0]);
					String predictionLabelsku = predictionMapSku.get(parts[0]);
					sum_abs=abs(predictionLabeluser-predictionLabelsku)+sum_abs
					sum_abs_square=abs(predictionLabeluser-predictionLabelsku)^2
					i++
				}
				sum_count=i
			}

			brStand.close();

			double mae = 0.0;
			if (sum_count!= 0) {
				mae = sum_abs / sum_count;
			}

			double mse = 0.0;
			if (sum_count != 0) {
				recall = sum_abs_square / sum_count;
			}

			System.out.println("mae:" + mae);
			System.out.println("mse:" + mse);
//			System.out.println("score:" + (0.4 * f1 + 0.6 * f2));
			double score = 0.4 * f1 + 0.6 * f2;
			System.out.println("score:" + score);
			result[2]= (float)score;
			result[4]= (float)mae;
			result[5]= (float)mse;
			return result;
		} catch (Exception e) {
			e.printStackTrace();
			result[0]=2;
			return result;
		}
	}

    public static void main(String[] args) {
      
        
    }

}
